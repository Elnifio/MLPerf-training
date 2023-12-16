import os
os.environ['DGL_PREFETCHER_TIMEOUT'] = str(600)

import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torch.multiprocessing as mp
import time, tqdm, numpy as np
from functools import partial
import json

from dllogger import Verbosity
from utility.feature_fetching import Features, IGBHeteroGraphStructure
from utility.components import build_graph, get_loader, RGAT
from utility.logger import IntegratedLogger

SEED = 0
import warnings
warnings.filterwarnings("ignore")

PAPER="paper" # we don't have any other labels to predict


def evaluate(dataloader, model, feature_store, device):
    epoch_start = time.time()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:

            batch_preds, batch_labels = model(batch, device, feature_store)

            labels.append(batch_labels.cpu().numpy())
            predictions.append(batch_preds.argmax(1).cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions) * 100
    return time.time() - epoch_start, accuracy


def run(
        proc_id, devices, backend, use_pyg_sampler, 
        graph, num_classes, in_memory, hp_config,
        modelpath, model_save, 
        fan_out, batch_size, num_workers, use_uva,
        hidden_channels, num_heads, dropout,
        learning_rate, decay, sched_stepsize, sched_gamma,
        epochs,
        train_idx, val_idx, 
        add_timer, no_debug, eval_frequency,
        feature_store):

    logger = IntegratedLogger(0, add_timer, no_debug=no_debug, print_only=True)

    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='13579')

    logger.debug("Initializing process group through torch.distributed.init_process_group")

    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
        
    world_size = len(devices)
    local_rank = proc_id

    logger.debug(f"Train val test data got. Initializing train dataloader through dgl.dataloading.DataLoader.")
    train_indices = train_idx
    val_indices = val_idx

    if use_uva: 
        train_indices = train_indices.to(device)
        val_indices = val_indices.to(device)

    train_dataloader = get_loader(
        graph=graph, 
        index=train_indices,
        fanouts=fan_out,
        backend=backend,
        use_pyg_sampler=use_pyg_sampler,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        num_workers=num_workers,
        use_uva=use_uva, 
        use_ddp=True, 
        ddp_seed=SEED
    )

    num_batches = len(train_dataloader)
    do_interval_eval = eval_frequency > 0
    if do_interval_eval:
        eval_interval = num_batches // eval_frequency + 1
        print(f"    [Rank {local_rank}] Total length of train dataloader: {len(train_dataloader)}; evaluating every {eval_interval} steps;")
    else:
        eval_interval = -1

    val_dataloader = get_loader(
        graph=graph, 
        index=val_indices,
        fanouts=fan_out,
        backend=backend,
        use_pyg_sampler=use_pyg_sampler,
        batch_size=batch_size,
        shuffle=True,
        device=device,
        num_workers=num_workers,
        use_uva=use_uva,
        use_ddp=True, 
        ddp_seed=SEED
    )

    if feature_store is not None and not in_memory:
        # if we use memory-mapped feature,
        # then we need to build the feature here
        # to avoid loading the features into the memory
        # else it's fine, and 
        # torch.multiprocessing.spawn helps us manage shared in-memory tensors 
        feature_store.build_features(g)

    in_feats = 1024

    logger.debug(f"Input features got. Defining model, optimizer, and scheduler.")

    model = RGAT(
        backend=backend, device=device,
        graph=graph,
        in_feats=in_feats,
        h_feats=hidden_channels,
        num_classes=num_classes,
        num_layers=len(fan_out.split(",")),
        n_heads=num_heads,
        dropout=dropout
    ).to(device)

    logger.debug(f"     Initialized model has {len(model.layers)} layers.")

    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=decay)

    sched = optim.lr_scheduler.StepLR(optimizer, step_size=sched_stepsize, gamma=sched_gamma)

    logger.debug(f"Model, optimizer, and scheduler initialized.")

    if proc_id == 0:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        logger.metadata("model_size", {"unit": "MB", "format": ":.3f"})
        hp_config['model_size'] = size_all_mb
        logger.log(step="PARAMETER", data=hp_config, verbosity=Verbosity.DEFAULT)
        # print('From proc 0: model size: {:.3f}MB'.format(size_all_mb))

    logger.metadata("gpu_memory", {"unit": "MB", "format": ":.2f"})
    logger.metadata("train_loss", {"unit": "", "format": ":.4f"})
    logger.metadata("train_acc", {"unit": "%", "format": ":.2f"})
    logger.metadata("valid_acc", {"unit": "%", "format": ":.2f"})
    logger.metadata("test__acc", {"unit": "%", "format": ":.2f"})
    logger.metadata("best_valid_acc", {"unit": "%", "format": ":.2f"})

    # Training loop
    collected_metrics = []
    for epoch in range(epochs):
        logger.debug(f"Starting epoch {epoch}.")
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        epoch_loss = 0
        epoch_start = time.time()
        logger.debug(f"Toggling model as train here.")
        model.train()
        train_losses = []
        accs = []
        eval_accs = []

        gpu_mem_alloc = []

        logger.debug(f"Making the train dataloader enumerable here")
        iterator = enumerate(train_dataloader)

        logger.debug("Starting to iterate through the dataloader")
        eval_counter = 0
        for step, batch in iterator:
            # in_epoch_eval_acc will have the same length as losses & train accs
            # this helps us correspond each in-epoch eval acc with the step
            eval_acc_for_this_step = -1
            if step % 3000 == 0 and add_timer:
                # for train time estimation
                print(f"Rank {local_rank} reached step {step} at {datetime.datetime.now()}")

            if do_interval_eval and step % eval_interval == 0:
                torch.cuda.synchronize()
                torch.distributed.barrier()
                eval_start = time.time()
                model.eval()
                val_time, eval_acc_for_this_step = evaluate(
                    dataloader = val_dataloader,
                    model=model,
                    feature_store=feature_store, 
                    device=device
                )
                eval_counter += 1
                torch.cuda.synchronize()
                torch.distributed.barrier()
                print(
                    f"Rank {local_rank}'s {eval_counter}-th eval before step {step}, at epoch {epoch}: {eval_acc_for_this_step}. "
                    +
                    (f"Eval time: {str(datetime.timedelta(seconds=int(time.time() - eval_start)))}" if add_timer else "")
                )
                model.train()

            eval_accs.append(eval_acc_for_this_step)

            batch_pred, batch_labels = model(batch, device, feature_store)
            batch_accuracy = sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(), batch_pred.argmax(1).detach().cpu().numpy()).item()*100
            
            loss = loss_fcn(batch_pred, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            train_losses.append(loss.detach().item())
            accs.append(batch_accuracy)
            gpu_mem_alloc.append(torch.cuda.max_memory_allocated(device=device) / 1000000 if torch.cuda.is_available() else 0)
        
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # end of a training epoch
        sched.step()        
        epoch_acc = sum(accs) / len(accs)
        epoch_gpu_mem = sum(gpu_mem_alloc) / len(gpu_mem_alloc)

        logger.log(
            step=(epoch, "train"),
            data={
                "rank": local_rank,
                "train_acc": epoch_acc,
                "train_time": str(datetime.timedelta(seconds=int(time.time() - epoch_start))),
                "train_loss": epoch_loss,
                "gpu_memory": epoch_gpu_mem, 
                "total_train_steps": len(accs),
            },
            verbosity=Verbosity.DEFAULT
        )

        torch.cuda.synchronize()
        torch.distributed.barrier()
        model.eval()
        val_time, val_acc = evaluate(
            dataloader=val_dataloader,
            model=model,
            feature_store=feature_store,
            device=device
        )
        torch.cuda.synchronize()
        torch.distributed.barrier()

        logger.log(
            step=(epoch, "valid"),
            data={
                "rank": local_rank,
                "valid_acc": val_acc,
                "valid_time": str(datetime.timedelta(seconds=int(val_time)))
            },
            verbosity=Verbosity.DEFAULT
        )

        if model_save:
            torch.save(model.state_dict(), modelpath)

        collected_metrics.append(
            {
                "rank": local_rank,
                "epoch": epoch,
                "train_losses": train_losses,
                "train_accs": accs,
                "eval_accs": eval_accs,
                "val_acc": val_acc
            }
        )

    # Export metrics for every rank
    metrics_dir="/results"

    def path_formatter(sampler_style, dataset_size, rank):
        trial_count = 0
        while os.path.exists(f"{metrics_dir}/{sampler_style}-sampler-{dataset_size}_hidden_{hidden_channels}_attn_{num_heads}_fanout_{fan_out}_rank_{rank}_trial_{trial_count}.json"):
            trial_count += 1
        return f"{metrics_dir}/{sampler_style}-sampler-{dataset_size}_hidden_{hidden_channels}_attn_{num_heads}_fanout_{fan_out}_rank_{rank}_trial_{trial_count}.json"
    
    with open(path_formatter(("pyg" if use_pyg_sampler else "dgl"), feature_store.dataset_size, local_rank), "w") as f:
        json.dump(collected_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/data', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument("--use_fp16", action="store_true", help="If present, use FP16 torch tensor input. Must have --in_memory 1")

    # Dataloader parameters 
    parser.add_argument('--fan_out', type=str, default='5,10,15')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--use_uva", action="store_true") # DGL Specific: use_uva for DGL graphs
    parser.add_argument("--separate_sampling_aggregation", action="store_true") # do we initialize the features separately, and not stored on graph? 
    parser.add_argument("--use_pyg_sampler", action="store_true")
    parser.add_argument("--backend", type=str, default="DGL")
    parser.add_argument("--layout", type=str, default="coo", choices=['coo', 'csr', 'csc'])

    # Model and Optimizer parameters
    parser.add_argument('--model_type', type=str, default='rgat',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='/workspace/model.pt')
    parser.add_argument('--model_save', type=int, default=0)

    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)

    # not active
    parser.add_argument("--sched_stepsize", type=int, default=25)
    parser.add_argument("--sched_gamma", type=float, default=0.25)

    # Training and logs related
    parser.add_argument('--in_epoch_eval_times', type=int, default=4)
    parser.add_argument('--gpu_devices', type=str, default='0,1,2,3,4,5,6,7')

    parser.add_argument("--add_timer", action="store_true")
    parser.add_argument("--no_debug", action="store_true")

    args = parser.parse_args()

    if args.use_uva: 
        args.num_workers = 0

    IntegratedLogger(
        proc_id=0, 
        add_timer=args.add_timer, 
        no_debug=args.no_debug
    ).debug(
        "Start creating graphs", 
        do_log=True
    )

    gpu_idx = [int(fanout) for fanout in args.gpu_devices.split(',')]
    num_gpus = len(gpu_idx)

    dataset = IGBHeteroGraphStructure(
        path=args.path,
        dataset_size=args.dataset_size,
        in_memory=args.in_memory,
        num_classes=args.num_classes,
        separate_sampling_aggregation=args.separate_sampling_aggregation,
        multithreading=True,
    )

    IntegratedLogger(
        proc_id=0, 
        add_timer=args.add_timer, 
        no_debug=args.no_debug
    ).debug(
        "Start loading features", 
        do_log=True
    )

    feature_store = Features(
        path=args.path, 
        dataset_size=args.dataset_size, 
        in_memory=args.in_memory, 
        use_fp16=args.use_fp16)

    feature_store.build_features(dataset.use_journal_conference)

    if args.in_memory:
        # we explicitly share the features here
        feature_store.share_features()

    g = build_graph(
        graph_structure=dataset,
        backend=args.backend,
        features=feature_store
    )

    if not args.separate_sampling_aggregation: 
        # If the features are already initialized to graph:
        # then we do not need to store another copy here
        feature_store.feature = {}

    if args.layout != "coo":
        # this can be performed offline
        g = g.formats(args.layout)

    IntegratedLogger(
        proc_id=0, 
        add_timer=args.add_timer, 
        no_debug=args.no_debug
    ).debug(
        "Start spawning processes", 
        do_log=True
    )

    mp.spawn(run, args=(
        gpu_idx, args.backend, args.use_pyg_sampler, 
        g, args.num_classes, args.in_memory, vars(args),
        args.modelpath, args.model_save,
        args.fan_out, args.batch_size, args.num_workers, args.use_uva,
        args.hidden_channels, args.num_heads, args.dropout,
        args.learning_rate, args.decay, args.sched_stepsize, args.sched_gamma,
        args.epochs,
        dataset.train_indices, dataset.val_indices,
        args.add_timer, args.no_debug, args.in_epoch_eval_times,
        feature_store
    ), nprocs=num_gpus)

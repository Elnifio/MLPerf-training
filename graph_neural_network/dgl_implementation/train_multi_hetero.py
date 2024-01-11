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

import pickle

from dllogger import Verbosity
from utility.feature_fetching import Features, IGBHeteroGraphStructure
from utility.components import build_graph, get_loader, RGAT
from utility.logger import IntegratedLogger

SEED = 0
import warnings
warnings.filterwarnings("ignore")

PAPER="paper" # we don't have any other labels to predict


def evaluate(dataloader, model, feature_store, device, eval_batches):
    epoch_start = time.time()
    predictions = []
    labels = []
    target_ids = []
    with torch.no_grad():
        for batch in dataloader:
            if eval_batches > 0 and len(predictions) > eval_batches:
                break

            batch_preds, batch_labels = model(batch, device, feature_store)
            if len(predictions) == 0:
                print(f"Rank 0 first 25 batch labels: {batch_labels[:25]}")

            labels.append(batch_labels.cpu().numpy())
            predictions.append(batch_preds.argmax(1).cpu().numpy())

            target_ids.append(batch[-1][-1].dstdata[dgl.NID]['paper'].cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        target_ids = np.concatenate(target_ids)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions) * 100
    return time.time() - epoch_start, accuracy, (predictions, labels, target_ids)


def run(
        proc_id, devices, backend, use_pyg_sampler, 
        graph, num_classes, in_memory, hp_config,
        modelpath, model_save, 
        fan_out, batch_size, num_workers, use_uva,
        hidden_channels, num_heads, 
        learning_rate, 
        epochs,
        train_idx, val_idx, 
        add_timer, no_debug, 
        validation_frac_within_epoch, in_epoch_eval_fraction, early_stop, target_accuracy,
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
    # train_indices = train_idx.split(train_idx.size(0) // world_size)[proc_id]
    # val_indices = val_idx.split(val_idx.size(0) // world_size)[proc_id]

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
    validation_freq = int(num_batches * validation_frac_within_epoch)

    is_success = False

    val_dataloader = get_loader(
        graph=graph, 
        index=val_indices,
        fanouts=fan_out,
        backend=backend,
        use_pyg_sampler=use_pyg_sampler,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        num_workers=num_workers,
        use_uva=use_uva,
        use_ddp=True,
        ddp_seed=SEED
    )

    num_eval_batches = int(len(val_dataloader) * in_epoch_eval_fraction)

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
        n_heads=num_heads
    ).to(device)

    logger.debug(f"     Initialized model has {len(model.layers)} layers.")

    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate)

    logger.debug(f"Model and optimizer initialized.")

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
    logger.metadata("rank_valid_acc", {"unit": "%", "format": ":.2f"})

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

            batch_pred, batch_labels = model(batch, device, feature_store)
            batch_accuracy = sklearn.metrics.accuracy_score(batch_labels.cpu().numpy(), batch_pred.argmax(1).detach().cpu().numpy()).item()*100
            
            loss = loss_fcn(batch_pred, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % validation_freq == 0:
                torch.cuda.synchronize()
                torch.distributed.barrier()
                eval_start = time.time()
                model.eval()
                all_reduced_val_acc = torch.tensor([0., ]).to(device)
                val_time, val_acc, validation_results = evaluate(
                    dataloader = val_dataloader,
                    model=model,
                    feature_store=feature_store, 
                    device=device,
                    eval_batches=num_eval_batches
                )
                eval_accs.append(val_acc)

                all_reduced_val_acc += val_acc
                torch.distributed.all_reduce(all_reduced_val_acc, op=torch.distributed.ReduceOp.SUM)
                all_reduced_val_acc /= world_size
                all_reduced_val_acc = all_reduced_val_acc.to("cpu").item()

                eval_counter += 1
                torch.cuda.synchronize()
                torch.distributed.barrier()
                print(
                    f"Rank {local_rank}'s {eval_counter}-th eval before step {step}, at epoch {epoch}: {val_acc}. All reduced val acc: {all_reduced_val_acc}. "
                    +
                    (f"Eval time: {str(datetime.timedelta(seconds=int(time.time() - eval_start)))}" if add_timer else "")
                )

                model.train()

                if early_stop and all_reduced_val_acc >= target_accuracy: 
                    is_success = True
                    break
            else:
                eval_accs.append(-1)

            epoch_loss += loss.detach().item()
            train_losses.append(loss.detach().item())
            accs.append(batch_accuracy)
            gpu_mem_alloc.append(torch.cuda.max_memory_allocated(device=device) / 1000000 if torch.cuda.is_available() else 0)
        
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # end of a training epoch
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

        if early_stop and is_success: 
            collected_metrics.append(
                {
                    "rank": local_rank,
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "train_accs": accs,
                    "eval_accs": eval_accs,
                    "val_acc": -1
                }
            )
        else:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            model.eval()
            all_reduced_val_acc = torch.tensor([0., ]).to(device)

            val_time, val_acc, _ = evaluate(
                dataloader=val_dataloader,
                model=model,
                feature_store=feature_store,
                device=device,
                eval_batches=num_eval_batches
            )

            all_reduced_val_acc += val_acc
            torch.cuda.synchronize()
            torch.distributed.barrier()

            torch.distributed.all_reduce(all_reduced_val_acc, op=torch.distributed.ReduceOp.SUM)
            all_reduced_val_acc /= world_size
            all_reduced_val_acc = all_reduced_val_acc.to("cpu").item()

            logger.log(
                step=(epoch, "valid"),
                data={
                    "rank": local_rank,
                    "rank_valid_acc": val_acc,
                    "valid_acc": all_reduced_val_acc, 
                    "valid_time": str(datetime.timedelta(seconds=int(val_time)))
                },
                verbosity=Verbosity.DEFAULT
            )

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

            if all_reduced_val_acc >= target_accuracy: 
                is_success = True
                break

    # Export metrics for every rank
    metrics_dir="/results"

    def path_formatter(rank):
        trial_count = 0
        while os.path.exists(f"{metrics_dir}/rank_{rank}_trial_{trial_count}.json"):
            trial_count += 1
        return f"{metrics_dir}/rank_{rank}_trial_{trial_count}.json"
    
    with open(path_formatter(local_rank), "w") as f:
        json.dump(collected_metrics, f)

    print(f"Rank {local_rank} train status: {is_success}")


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
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)

    # Training and logs related
    parser.add_argument('--validation_frac_within_epoch', type=float, default=0.2)
    parser.add_argument('--in_epoch_eval_fraction', type=float, default=0.05)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--target_accuracy", type=float, default=72.0)

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
        args.hidden_channels, args.num_heads, 
        args.learning_rate, 
        args.epochs,
        dataset.train_indices, dataset.val_indices,
        args.add_timer, args.no_debug, 
        args.validation_frac_within_epoch, args.in_epoch_eval_fraction, args.early_stop, args.target_accuracy,
        feature_store
    ), nprocs=num_gpus)

from data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
    Dataset_Solar, Dataset_PEMS, Dataset_Pred
)
from data_provider.data_blast import Dataset_BLAST
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'BLAST': Dataset_BLAST,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    f = flag.lower()

    timeenc = 0 if getattr(args, 'embed', None) != 'timeF' else 1

    if args.data == 'BLAST':
        split_flag = 'train' if f == 'train' else 'val'
        shuffle_flag = (split_flag == 'train')
        drop_last = False
        batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            flag=split_flag,
            size=[args.seq_len, args.pred_len],
            scale=True,
        )
    elif args.data == 'GlobalTemp':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            context_length=int(args.seq_len),
            prediction_length=int(args.pred_len),
        )
    else:
        shuffle_flag = False if (f in ('test', 'test'.upper())) else True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

        std_flag = 'test' if f in ('test', 'test'.upper()) else flag

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=std_flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1)
    return data_set, data_loader

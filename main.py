"""New framework CLI. Input .pt dictionaries contain tensors (see README)."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from models.latent_completion import MissingRegionAutoencoder, LatentCompletion, extract_missing_region
from run.stages import autoencoder_step, flow_step


class TensorSamples(Dataset):
    def __init__(self, data, stage, threshold=0.03, chunk_size=512):
        self.data = data
        self.stage = stage
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.generate_missing = stage != 'sample' and 'missing' not in data
        keys = ('missing',) if stage == 'ae' else ('partial', 'image', 'missing')
        if stage == 'sample':
            keys = ('partial', 'image')
        if self.generate_missing:
            keys = tuple(dict.fromkeys(tuple(k for k in keys if k != 'missing') + ('gt', 'partial')))
        for key in keys:
            if key not in data or not torch.is_tensor(data[key]):
                raise ValueError('Missing tensor: ' + key)
        if len({data[key].shape[0] for key in keys}) != 1:
            raise ValueError('Batch counts differ')
        self.keys = keys

    def __len__(self):
        return self.data[self.keys[0]].shape[0]

    def __getitem__(self, index):
        sample = {key: self.data[key][index] for key in self.keys}
        if self.generate_missing:
            try:
                sample['missing'] = extract_missing_region(
                    sample['gt'][None], sample['partial'][None],
                    threshold=self.threshold, chunk_size=self.chunk_size)[0]
            except ValueError as error:
                raise ValueError(f'Dataset sample {index}: {error}') from error
            sample.pop('gt')
            if self.stage == 'ae':
                sample.pop('partial')
        return sample


def main():
    parser = argparse.ArgumentParser(description='Missing-region latent flow completion')
    parser.add_argument('--stage', choices=('ae', 'flow', 'sample'), required=True)
    parser.add_argument('--data', required=True, help='Tensor dictionary .pt')
    parser.add_argument('--ae-checkpoint')
    parser.add_argument('--checkpoint', help='Resume stage checkpoint or load flow for sampling')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.03)
    parser.add_argument('--missing-chunk-size', type=int, default=512,
                        help='Max points per axis in missing-region distance blocks')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.missing_chunk_size < 1:
        parser.error('epochs, batch-size and missing-chunk-size must be positive')
    data = torch.load(args.data, map_location='cpu', weights_only=True)
    dataset = TensorSamples(data, args.stage, args.threshold, args.missing_chunk_size)
    if not len(dataset):
        parser.error('empty dataset')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.stage != 'sample')
    resume = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if resume and resume['stage'] != ('flow' if args.stage == 'sample' else args.stage):
        parser.error('checkpoint belongs to a different stage')
    ae = MissingRegionAutoencoder()
    if args.stage != 'ae':
        if resume is None:
            if not args.ae_checkpoint:
                parser.error('flow requires --ae-checkpoint (or a resumed flow checkpoint)')
            checkpoint = torch.load(args.ae_checkpoint, map_location='cpu', weights_only=True)
            if checkpoint['stage'] != 'ae':
                parser.error('--ae-checkpoint must contain a pretrained autoencoder')
            ae.load_state_dict(checkpoint['model'])
        model = LatentCompletion(ae)
    else:
        model = ae
    if resume:
        model.load_state_dict(resume['model'])
    model.to(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'sample':
        if resume is None:
            parser.error('sample requires --checkpoint from flow training')
        results = []
        for batch in loader:
            results.append({k: v.cpu() for k, v in model.sample(
                batch['partial'].to(args.device), batch['image'].to(args.device), args.steps).items()})
        torch.save({k: torch.cat([result[k] for result in results]) for k in results[0]}, args.output)
        return
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    start = 0
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        start = resume['epoch']
    for epoch in range(start, args.epochs):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = (autoencoder_step(model, batch['missing'], optimizer) if args.stage == 'ae'
                    else flow_step(model, batch, optimizer))
            total += loss.item() * batch['missing'].shape[0]
        print(f'Epoch {epoch + 1}: loss={total / len(dataset):.6f}')
        torch.save({'stage': args.stage, 'epoch': epoch + 1, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, args.output)


if __name__ == '__main__':
    main()

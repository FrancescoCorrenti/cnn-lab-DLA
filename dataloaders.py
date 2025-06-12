import torch
from torch.utils.data import DataLoader
class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, subset, teacher_outputs):
        # If `subset` is a torch.utils.data.Subset, extract its indices & underlying dataset
        if hasattr(subset, 'indices') and hasattr(subset, 'dataset'):
            self.indices = list(subset.indices)
            self.dataset = subset.dataset
        else:
            # Otherwise assume it’s a “raw” Dataset
            self.indices = list(range(len(subset)))
            self.dataset = subset

        # Ensure teacher_outputs is a torch.Tensor of shape [N, ...]
        if not isinstance(teacher_outputs, torch.Tensor):
            teacher_outputs = torch.as_tensor(teacher_outputs)
        self.teacher_outputs = teacher_outputs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        data, target = self.dataset[orig_idx]
        teacher_logits = self.teacher_outputs[orig_idx].clone().detach()
        return data, target, teacher_logits
 

def get_distill_loaders(trainset, teacher_outputs_pt, batch_size, val_percent, num_workers, pin_memory):
    # carica teacher_outputs in GPU
    teacher_outputs = torch.load(teacher_outputs_pt, map_location="cpu")
    full_distill = DistillDataset(trainset, teacher_outputs)

    total = len(full_distill)
    split = int((1 - val_percent) * total)
    perm = torch.randperm(total).tolist()
    train_ids, val_ids = perm[:split], perm[split:]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_sampler   = torch.utils.data.SubsetRandomSampler(val_ids)

    train_loader = DataLoader(full_distill,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(full_distill,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader
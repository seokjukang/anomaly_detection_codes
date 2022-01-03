import torch


class Config:
    def __init__(self):
        self.isCudaAvailable = False
        self.device = self._get_device()

    def _get_device(self):
        print(torch.__version__)

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        if device.type == 'cuda':
            self.isCudaAvailable = True
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

        try:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            return None

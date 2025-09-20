import os
import yaml
from pathlib import Path
import torch


class Config:
    def __init__(self, config_path=None):
        current_folder = Path(__file__).parent
        config_path = os.path.join(current_folder, "config.yaml") if not config_path else config_path
        self._path = config_path

        with open(config_path, "r") as f:  # load entire cfg
            self.cfg = yaml.safe_load(f)

        self.root_dir = Path(__file__).parents[2]  # root folder is two levels above
        self.data_dir = self.root_dir / self.cfg["paths"]["data_dir"]

        self.dataset_raw = self.root_dir / self.cfg["paths"]["dataset_raw"]
        self.train_data = self.root_dir / self.cfg["paths"]["train_data"]
        self.test_data = self.root_dir / self.cfg["paths"]["test_data"]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: add frequently used fields as class attributes

    def __getitem__(self, item: str):
        return self.cfg[item]
    
    def save_config(self, config_path=None):
        """
        Write the current configuration to a YAML file.
        Args:
            config_path (str): Path to the YAML file. If None, the default path is used.
        """
        config_path = self._path if not config_path else config_path
        with open(config_path, "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)

# if imported as a module, this becomes my config singleton
CFG = Config()

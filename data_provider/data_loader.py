import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class UnivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        
    def __getitem__(self, index):
        feat_id = index // self.n_timepoint
        s_begin = index % self.n_timepoint
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
                
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_var * self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class LoginIntervalUser(Dataset):
    def __init__(
        self,
        root_path,
        flag='train',
        size=None,
        data_path='user.csv',
        scale=True,
        nonautoregressive=False,
        test_flag='T',
        subset_rand_ratio=1.0,
        target_col='y_delta_next_min',
        time_col='date',
        feature_cols=None,
        drop_cols='user_id',
        append_target_channel=True,
    ):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        self.target_col = target_col
        self.time_col = time_col
        self.feature_cols = None if feature_cols in [None, ''] else [
            c.strip() for c in feature_cols.split(',') if c.strip()
        ]
        if isinstance(drop_cols, str):
            self.drop_cols = [c.strip()
                              for c in drop_cols.split(',') if c.strip()]
        else:
            self.drop_cols = list(drop_cols)
        self.append_target_channel = append_target_channel
        self.__read_data__()

    def __read_data__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(dataset_file_path)

        if self.time_col in df_raw.columns:
            df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col])
            df_raw = df_raw.sort_values(self.time_col).reset_index(drop=True)

        if self.target_col not in df_raw.columns:
            raise ValueError(
                f"target_col '{self.target_col}' not found in {dataset_file_path}")

        if self.feature_cols is None:
            candidate_cols = df_raw.columns.tolist()
            drop_set = set(self.drop_cols + [self.target_col])
            if self.time_col in candidate_cols:
                drop_set.add(self.time_col)
            feature_cols = []
            for c in candidate_cols:
                if c in drop_set:
                    continue
                if pd.api.types.is_numeric_dtype(df_raw[c]):
                    feature_cols.append(c)
            if not feature_cols:
                raise ValueError(
                    "No numeric feature columns found for LoginIntervalUser")
            self.feature_cols = feature_cols
        else:
            missing = [c for c in self.feature_cols if c not in df_raw.columns]
            if missing:
                raise ValueError(
                    f"feature_cols missing in data: {missing}")

        x_all = df_raw[self.feature_cols].values.astype(np.float32)
        y_all = df_raw[[self.target_col]].values.astype(np.float32)
        if self.time_col in df_raw.columns:
            ts_all = df_raw[self.time_col].dt.strftime(
                "%Y-%m-%d %H:%M:%S").values
        else:
            ts_all = np.array([str(i) for i in range(len(df_raw))])
        idx_all = np.arange(len(df_raw))

        data_len = len(df_raw)
        num_train = int(data_len * 0.7)
        num_val = int(data_len * 0.15)
        num_test = data_len - num_train - num_val
        train_start, train_end = 0, num_train
        val_start, val_end = train_end, train_end + num_val
        test_start, test_end = val_end, data_len

        if self.scale:
            self.scaler_x.fit(x_all[train_start:train_end])
            self.scaler_y.fit(y_all[train_start:train_end])
            x_all = self.scaler_x.transform(x_all)
            y_all = self.scaler_y.transform(y_all)

        if self.append_target_channel:
            target_zeros = np.zeros_like(y_all)
            x_all = np.concatenate([x_all, target_zeros], axis=1)
            y_all = np.concatenate([np.zeros_like(x_all[:, :-1]), y_all], axis=1)

        if self.set_type == 0:
            start, end = train_start, train_end
        elif self.set_type == 1:
            start, end = val_start, val_end
        else:
            start, end = test_start, test_end

        self.data_x = x_all[start:end]
        self.data_y = y_all[start:end]
        self.timestamps = ts_all[start:end]
        self.row_indices = idx_all[start:end]

        if len(self.data_x) < self.seq_len:
            raise ValueError(
                f"Not enough data in split '{self.flag}' for seq_len={self.seq_len}")

        self.sample_indices = np.arange(self.seq_len - 1, len(self.data_x))

    def __getitem__(self, index):
        t = self.sample_indices[index]
        s_begin = t - self.seq_len + 1
        s_end = t + 1
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[t:t + self.output_token_len]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        row_idx = int(self.row_indices[t])
        timestamp = self.timestamps[t]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, row_idx, timestamp

    def __len__(self):
        return len(self.sample_indices)

    def inverse_transform_y(self, data):
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler_y.inverse_transform(data)

    def inverse_transform_x(self, data):
        return self.scaler_x.inverse_transform(data)


class MultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
            
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Global_Temp(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "temp_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Global_Wind(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "wind_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Dataset_ERA5_Pretrain(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        Train_S = int(S * 0.8)
        df_raw = df_raw[:, :Train_S]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ERA5_Pretrain_Test(Dataset):
    def __init__(self, root_path, flag='test', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.test_flag = test_flag
        assert test_flag in ['T', 'V', 'TandV']
        type_map = {'T': 0, 'V': 1, 'TandV': 2}
        self.test_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        if self.test_type == 0:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, :Train_S]
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len,
                        len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            data = df_raw
            border1 = border1s[-1]
            border2 = border2s[-1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
        else:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, Train_S:]
            num_train = int(len(df_raw) * 0.8)
            num_test = len(df_raw) - num_train
            border1s = [0, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, len(df_raw)]
            data = df_raw
            if self.test_type == 1:
                border1 = border1s[0]
                border2 = border2s[0]
            else:
                border1 = border1s[1]
                border2 = border2s[1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Download link: https://huggingface.co/datasets/thuml/UTSD
class UTSD(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.root_path = root_path
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.csv'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    df_raw = pd.read_csv(dataset_path)

                    if isinstance(df_raw[df_raw.columns[0]][0], str):
                        data = df_raw[df_raw.columns[1:]].values
                    else:
                        data = df_raw.values

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)
                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]


# Download link: https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
class UTSD_Npy(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.npy'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    data = np.load(dataset_path)

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)

                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]

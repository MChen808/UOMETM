from torch.utils import data
import torch
import csv
import h5py


class Data_preprocess_ADNI:
    def __init__(self, number=0.25, label=-1):
        self.number = number
        self.label = label

        # demographic
        # self.demo_train = h5py.File('./ADNI/adni_all_surf_info_regular_longitudinal_random_train.mat')
        # self.demo_test = h5py.File('./ADNI/adni_all_surf_info_regular_longitudinal_random_test.mat')
        self.demo_train = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
                                    'adni_all_surf_info_regular_longitudinal_random_train.mat')
        self.demo_test = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
                                   'adni_all_surf_info_regular_longitudinal_random_test.mat')
        print('Reading demographical data finished...')

        # thickness
        self.thickness_train = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
                                         'adni_all_surf_thickness_regular_longitudinal_random_train.mat')
        self.thickness_test = h5py.File('/projects/students/chaoqiang/VGCNNRNN/DataPrepare/DataOutput/'
                                        'adni_all_surf_thickness_regular_longitudinal_random_test.mat')
        print('Reading thickness data finished...')

        # sort index
        self.label_idx_train, self.label_idx_test = None, None
        self.idx_train, self.idx_test = None, None

    def generate_demo_train_test(self, fold):
        # load data
        age_train = torch.tensor(self.demo_train['Age']).float().squeeze()
        age_test = torch.tensor(self.demo_test['Age']).float().squeeze()
        label_train = torch.tensor(self.demo_train['Label']).float().squeeze()
        label_test = torch.tensor(self.demo_test['Label']).float().squeeze()
        timepoint_train = torch.tensor(self.demo_train['Wave']).float().squeeze()
        timepoint_test = torch.tensor(self.demo_test['Wave']).float().squeeze()
        with open('ADNI/subject_train.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_train = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()
        with open('ADNI/subject_test.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_test = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()

        # select label
        if self.label != -1:
            if self.label == 0:
                self.label_idx_train = torch.nonzero(label_train == 0)
                self.label_idx_test = torch.nonzero(label_test == 0)
            if self.label == 1:
                self.label_idx_train = torch.cat((torch.nonzero(label_train == 1), torch.nonzero(label_train == 2)), dim=0)
                self.label_idx_test = torch.cat((torch.nonzero(label_test == 1), torch.nonzero(label_test == 2)), dim=0)
            if self.label == 2:
                self.label_idx_train = torch.nonzero(label_train == 3)
                self.label_idx_test = torch.nonzero(label_test == 3)
            age_train, age_test = age_train[self.label_idx_train].squeeze(), age_test[self.label_idx_test].squeeze()
            label_train, label_test = label_train[self.label_idx_train].squeeze(), label_test[self.label_idx_test].squeeze()
            timepoint_train, timepoint_test = timepoint_train[self.label_idx_train].squeeze(), timepoint_test[self.label_idx_test].squeeze()
            subject_train, subject_test = subject_train[self.label_idx_train].squeeze(), subject_test[self.label_idx_test].squeeze()

        baseline_age_train, baseline_age_test = [], []
        s_old = None
        for age, subject in zip(age_train, subject_train):
            if s_old is None:
                baseline_age_train.append(age)
                s_old = subject
            else:
                if subject == s_old:
                    baseline_age_train.append(baseline_age_train[-1])
                else:
                    baseline_age_train.append(age)
                    s_old = subject
        for age, subject in zip(age_test, subject_test):
            if s_old is None:
                baseline_age_test.append(age)
                s_old = subject
            else:
                if subject == s_old:
                    baseline_age_test.append(baseline_age_test[-1])
                else:
                    baseline_age_test.append(age)
                    s_old = subject
        baseline_age_train = torch.tensor(baseline_age_train).float()
        baseline_age_test = torch.tensor(baseline_age_test).float()

        demo_train = {'age': age_train, 'baseline_age': baseline_age_train, 'label': label_train,
                      'subject': subject_train, 'timepoint': timepoint_train}
        demo_test = {'age': age_test, 'baseline_age': baseline_age_test, 'label': label_test,
                     'subject': subject_test, 'timepoint': timepoint_test}

        print('Generating demographical data finished...')
        if fold == 0:
            return demo_train, demo_test
        else:
            return demo_test, demo_train

    def generate_thick_train_test(self, fold):
        left_thick_train = self.thickness_train['lthick_regular'][:, :self.number]
        right_thick_train = self.thickness_train['rthick_regular'][:, :self.number]
        left_thick_test = self.thickness_test['lthick_regular'][:, :self.number]
        right_thick_test = self.thickness_test['rthick_regular'][:, :self.number]

        if self.label != -1:
            label_idx_train = self.label_idx_train.view(1, -1).squeeze().numpy()
            label_idx_test = self.label_idx_test.view(1, -1).squeeze().numpy()
            left_thick_train, left_thick_test = left_thick_train[label_idx_train], left_thick_test[label_idx_test]
            right_thick_train, right_thick_test = right_thick_train[label_idx_train], right_thick_test[label_idx_test]

        thick_train = {'left': left_thick_train, 'right': right_thick_train}
        thick_test = {'left': left_thick_test, 'right': right_thick_test}

        print('Generating thickness data finished...')
        if fold == 0:
            return thick_train, thick_test, self.number
        else:
            return thick_test, thick_train, self.number

    def generate_XY(self, data):
        N = data['age'].size()[0]
        I = len(torch.unique(data['subject']))

        delta_age = (data['age'] - data['baseline_age']).view(N, -1)
        ones = torch.ones(size=delta_age.size())
        X = torch.cat((ones, delta_age, data['baseline_age'].view(N, -1)), dim=1)

        Y, old_s, cnt_zero = None, None, 0
        for i in range(N):
            if old_s is None:
                old_s = data['subject'][i]
            elif old_s != data['subject'][i]:
                old_s = data['subject'][i]
                cnt_zero += 1

            zeros0 = torch.zeros(size=[1, 2 * cnt_zero])
            zeros1 = torch.zeros(size=[1, 2 * (I - 1 - cnt_zero)])
            yy = X[i, :2].view(1, 2)
            yy = torch.cat((zeros0, yy, zeros1), dim=1)

            if Y is None:
                Y = yy
            else:
                Y = torch.cat((Y, yy), dim=0)

        print('Generating X and Y finished...')
        return X, Y

    def generate_orig_data(self):
        # load data
        age_train = torch.tensor(self.demo_train['Age']).float().squeeze()
        age_test = torch.tensor(self.demo_test['Age']).float().squeeze()
        label_train = torch.tensor(self.demo_train['Label']).float().squeeze()
        label_test = torch.tensor(self.demo_test['Label']).float().squeeze()
        timepoint_train = torch.tensor(self.demo_train['Wave']).float().squeeze()
        timepoint_test = torch.tensor(self.demo_test['Wave']).float().squeeze()
        with open('ADNI/subject_train.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_train = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()
        with open('ADNI/subject_test.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            subject_test = torch.tensor([[int(cell[:3] + cell[6:]) for cell in row] for row in csvreader]).squeeze()

        demo_train = {'age': age_train, 'label': label_train,
                      'subject': subject_train, 'timepoint': timepoint_train}
        demo_test = {'age': age_test, 'label': label_test,
                     'subject': subject_test, 'timepoint': timepoint_test}
        print("Generating original demographical data finished...")

        left_thick_train = self.thickness_train['lthick_regular']
        right_thick_train = self.thickness_train['rthick_regular']
        left_thick_test = self.thickness_test['lthick_regular']
        right_thick_test = self.thickness_test['rthick_regular']

        thick_train = {'left': left_thick_train, 'right': right_thick_train}
        thick_test = {'left': left_thick_test, 'right': right_thick_test}

        print("Generating original thickness data finished...")

        return demo_train, demo_test, thick_train, thick_test


class Dataset_adni(data.Dataset):
    def __init__(self, left_thickness, right_thickness, age, baseline_age, label, subject, timepoint):
        self.lthick = left_thickness
        self.rthick = right_thickness
        self.age = age
        self.baseline_age = baseline_age
        self.label = label
        self.subject = subject
        self.timepoint = timepoint

    def __len__(self):
        return len(self.age)

    def __getitem__(self, index):
        a = self.lthick[index]
        b = self.rthick[index]
        c = self.age[index]
        d = self.baseline_age[index]
        e = self.label[index]
        f = self.subject[index]
        g = self.timepoint[index]
        return a, b, c, d, e, f, g
import csv


class CSVLogger():
    def __init__(self, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def writerow(self, epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc,
                 hypergradient_cos_diff, hypergradient_l2_diff, run_time, iteration):
        self.writer.writerow({
            'epoch': str(epoch),
            'train_loss': str(train_loss),
            'train_acc': str(train_acc),
            'val_loss': str(val_loss),
            'val_acc': str(val_acc),
            'test_loss': str(test_loss),
            'test_acc': str(test_acc),
            'hypergradient_cos_diff': hypergradient_cos_diff,
            'hypergradient_l2_diff': hypergradient_l2_diff,
            'run_time': run_time,
            'iteration': iteration
        })
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

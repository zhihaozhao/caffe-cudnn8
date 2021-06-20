## UCF-101 training demo

Follow the steps in [README.md](/#ucf-101-training-demo)

## Files in this directory

* `train_ucf101.sh`: a main script to run for training C3D on UCF-101 data
* `c3d_ucf101_solver.prototxt`: a solver specifications -- SGD parameters, testing parametesr, etc
* `c3d_ucf101_test_split1.txt`, `c3d_ucf101_train_split1.txt`: lists of testing/training video clips in ("video directory", "starting frame num", "label") format
* `c3d_ucf101_train_test.prototxt`: training/testing network model
* `ucf101_train_mean.binaryproto`: a mean cube calculated from UCF101 training set
* `c3d_ucf101_train_loss_accuracy.png`: a sample plot of training iteration vs loss and accuracy

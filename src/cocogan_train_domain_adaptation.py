#!/usr/bin/env python
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys
from tools import *
from trainers import *
from common import *
import torchvision
import itertools
import tensorboard
from tensorboard import summary as summary
from optparse import OptionParser
from torchsummary import summary as model_summary
from data_utils import DataHandler
parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--config',
                  type=str,
                  help="net configuration",
                  default="../exps/unit/svhn2mnist.yaml")
parser.add_option('--log',
                  type=str,
                  help="log path",
                  default="../logs/unit")

MAX_EPOCHS = 100000

def main(argv):
  (opts, args) = parser.parse_args(argv)

  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  batch_size = config.hyperparameters['batch_size']
  print(batch_size)
  max_iterations = config.hyperparameters['max_iterations']

  cmd = "trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer']
  local_dict = locals()
  exec(cmd,globals(),local_dict)
  trainer = local_dict['trainer']
  trainer.cuda(opts.gpu)

  iterations = 0

  train_writer = tensorboard.FileWriter("%s/%s" % (opts.log,os.path.splitext(os.path.basename(opts.config))[0]))
  snapshot_directory = prepare_snapshot_folder(config.snapshot_prefix)
  image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, iterations, config.image_save_iterations)

  # # Load datasets
  # train_loader_a = get_data_loader_svhn(config.datasets['train_a'], batch_size)
  # train_loader_b = get_data_loader_mnist(config.datasets['train_b'], batch_size)
  # test_loader_b = get_data_loader_mnist_test(config.datasets['test_b'], batch_size = config.hyperparameters['test_batch_size'])
  # print(train_loader_a)

  # load the data
  data = DataHandler()
  # svhn
  (svhn_train_x, svhn_train_y), \
  (svhn_test_x, svhn_test_y), (svhn_ext_x, svhn_ext_y), (_, _) \
          = data.svhn_ufldl("../datasets/svhn/")
  # svhn_train_x = data.resize(svhn_train_x, (28, 28))
  # svhn_test_x = data.resize(svhn_test_x, (28, 28))
  svhn_train_x = svhn_train_x.transpose(0, 3, 1, 2)
  svhn_test_x = svhn_test_x.transpose(0, 3, 1, 2)
  svhn_ext_x = svhn_ext_x.transpose(0, 3, 1, 2)

  # mnist
  (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = \
          data.mnist("../../../datasets/mnist-tf")
  mnist_train_x = data.resize(mnist_train_x, (32, 32))
  mnist_test_x = data.resize(mnist_test_x, (32, 32))

  mnist_train_x = mnist_train_x.transpose(0, 3, 1, 2)
  mnist_test_x = mnist_test_x.transpose(0, 3, 1, 2)

  print(svhn_train_x.shape, svhn_train_y.shape, svhn_test_x.shape,
        svhn_test_y.shape)
  size_batch = 64
  svhn_generator_tr = data.batch_generator([svhn_train_x/255.0,
                                           svhn_train_y], size_batch,
                                          shuffle=True)
  svhn_generator_ex = data.batch_generator([svhn_ext_x/255.0,
                                           svhn_ext_y], size_batch,
                                          shuffle=True)
  svhn_generator_te = data.batch_generator([svhn_test_x/255.0,
                                           svhn_test_y], size_batch,
                                          shuffle=True)

  mnist_generator_tr = data.batch_generator([mnist_train_x / 255.0,
                                             mnist_train_y], size_batch,
                                            shuffle=True)
  mnist_generator_te = data.batch_generator([mnist_test_x / 255.0,
                                            mnist_test_y], size_batch,
                                           shuffle=True)

  source_ge_tr = svhn_generator_tr
  target_ge_tr = mnist_generator_tr
  target_ge_te = mnist_generator_te

  best_score = 0
  for ep in range(0, MAX_EPOCHS):
    print("epoch:", ep)
    for it, ((images_a, labels_a), (images_b,labels_b)) in \
    enumerate(zip(source_ge_tr, target_ge_tr)):
      print("images:", it)
      # if images_a.size(0) != batch_size or images_b.size(0) != batch_size:
        # continue
      images_a = torch.tensor(images_a).float()
      labels_a = torch.LongTensor([np.int64(labels_a)])
      images_b = torch.tensor(images_b).float()
      labels_b = torch.tensor([np.int64(labels_b)])
      trainer.dis.train()
      images_a = Variable(images_a.cuda(opts.gpu))
      print("tensor images:", images_b)
      labels_a = Variable(labels_a.cuda(opts.gpu)).view(images_a.size(0))
      images_b = Variable(images_b.cuda(opts.gpu))
      # Main training code
      trainer.dis_update(images_a, labels_a, images_b, config.hyperparameters)
      x_aa, x_ba, x_ab, x_bb = trainer.gen_update(images_a, images_b, config.hyperparameters)

      # Dump training stats in log file
      if (iterations+1) % config.display == 0:
        write_loss(iterations, max_iterations, trainer, train_writer)

      # # Save network weights
      if (iterations+1) % config.snapshot_save_iterations == 0:
        trainer.dis.eval()
        score = 0
        num_samples = 0
        for tit, (test_images_b, test_labels_b) in enumerate(target_ge_te):
          print("test:", tit)
          test_images_b = torch.tensor(test_images_b).float()
          test_labels_b = torch.LongTensor([np.int64(test_labels_b)])
          print("test tensor images", test_images_b)
          test_images_b = Variable(test_images_b.cuda(opts.gpu))
          test_labels_b = Variable(test_labels_b.cuda(opts.gpu)).view(test_images_b.size(0))
          cls_outputs = trainer.dis.classify_b(test_images_b)
          _, cls_predicts = torch.max(cls_outputs.data, 1)
          cls_acc = (cls_predicts == test_labels_b.data).sum()
          score += cls_acc
          num_samples += test_images_b.size(0)
          if tit == 100:
              break
        score /= 1.0 * num_samples
        print(score)
        print('Classification accuracy for Test_B dataset: %4.4f' % score)
        if score > best_score:
          best_score = score
          trainer.save(config.snapshot_prefix, iterations=-1)
        train_writer.add_summary(summary.scalar('test_b_acc', score), iterations + 1)
        img_name = image_directory + "/images_a.jpg"
        torchvision.utils.save_image(images_a.data / 2 + 0.5, img_name)
        img_name = image_directory + "/images_b.jpg"
        torchvision.utils.save_image(images_b.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_aa.jpg"
        torchvision.utils.save_image(x_aa.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_ab.jpg"
        torchvision.utils.save_image(x_ab.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_bb.jpg"
        torchvision.utils.save_image(x_bb.data / 2 + 0.5, img_name)
        img_name = image_directory + "/x_ba.jpg"
        torchvision.utils.save_image(x_ba.data / 2 + 0.5, img_name)


      iterations += 1
      if iterations == max_iterations:
        return

if __name__ == '__main__':
  print(sys.argv)
  main(sys.argv)


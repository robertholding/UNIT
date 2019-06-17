import sys
from tools import *
from trainers import *
from common import *
import torchvision
import itertools
import tensorboard
from tensorboard import summary
from optparse import OptionParser
from torchsummary import summary
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
  config = NetConfig(opts.config)
  trainer = cocogan_trainer_da.COCOGANDAContextTrainer(config.hyperparameters)
  trainer1 = cocogan_nets_da.CoDis32x32()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = trainer1.to(device)
  summary(model, [(3, 32, 32), (1, 32, 32)])


if __name__ == '__main__':
  main(sys.argv)

# def main(argv):
  # (opts, args) = parser.parse_args(argv)

  # # Load experiment setting
  # assert isinstance(opts, object)
  # config = NetConfig(opts.config)

  # batch_size = config.hyperparameters['batch_size']
  # print(batch_size)
  # max_iterations = config.hyperparameters['max_iterations']

  # cmd = "trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer']
  # local_dict = locals()
  # exec(cmd,globals(),local_dict)
  # trainer = local_dict['trainer']
  # trainer.cuda(opts.gpu)

  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # model = trainer().to(device)
  # summary(model, (3, 28, 28))


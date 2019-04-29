import os
import torch
import torch.nn as nn
import argparse
from sklearn import metrics
from torch.utils.data import DataLoader
from models import LSTM, AE_LSTM, ATAE_LSTM, PBAN
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']], 
                max_length=opt.max_length, 
                data_file='{0}_tokenizer.dat'.format(opt.dataset))
        embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab, 
                embed_dim=opt.embed_dim, 
                data_file='{0}d_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
        trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, target_dim=self.opt.polarities_dim)
        testset = SentenceDataset(opt.dataset_file['test'], tokenizer, target_dim=self.opt.polarities_dim)
        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        if opt.device.type == 'cuda':
            print('cuda memory allocated:', torch.cuda.memory_allocated(self.opt.device.index))
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 50)
            print('epoch:', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = './state_dict/{0}_{1}_{2}class_acc{3:.4f}'.format(self.opt.model_name, self.opt.dataset, self.opt.polarities_dim, test_acc)
                            torch.save(self.model.state_dict(), path)
                            print('model saved:', path)
                    if f1 > max_f1:
                        max_f1 = f1
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1
    
    def _evaluate(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_dataloader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)
                
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0) if t_targets_all is not None else t_targets
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0) if t_outputs_all is not None else t_outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1
    
    def run(self, repeats=1):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            print('repeat:', i)
            self._reset_params()
            max_test_acc, max_f1 = self._train(criterion, optimizer, max_test_acc_overall)
            print('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            print('#' * 50)
        print('max_test_acc_overall:', max_test_acc_overall)
        print('max_f1_overall:', max_f1_overall)

def main():
    
    model_classes = {
        'lstm': LSTM,
        'ae_lstm': AE_LSTM,
        'atae_lstm': ATAE_LSTM,
        'pban': PBAN
    }
    
    dataset_files = {
        'restaurant': {
            'train': './datasets/Restaurants_Train.xml',
            'test': './datasets/Restaurants_Test.xml'
        },
        'laptop': {
            'train': './datasets/Laptops_Train.xml',
            'test': './datasets/Laptops_Test.xml'
        }
    }
    
    input_colses = {
        'lstm': ['text'],
        'ae_lstm': ['text', 'aspect'],
        'atae_lstm': ['text', 'aspect'],
        'pban': ['text', 'aspect', 'position']
    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,    # default lr=0.01
        'adam': torch.optim.Adam,          # default lr=0.001
        'adamax': torch.optim.Adamax,      # default lr=0.002
        'asgd': torch.optim.ASGD,          # default lr=0.01
        'rmsprop': torch.optim.RMSprop,    # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='pban', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--position_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int, help='2, 3')
    parser.add_argument('--max_length', default=80, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--repeats', default=1, type=int)
    opt = parser.parse_args()
    	
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    
    ins = Instructor(opt)
    ins.run(opt.repeats)

if __name__ == '__main__':
    main()

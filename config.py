import argparse  

def get_options(parser=argparse.ArgumentParser()):  
  
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out_dir', type=str, default='/kaggle/working/outputs')
    parser.add_argument('--output',action='store_true',default=True,help="shows output")

    
    opt = parser.parse_args()
    
    if opt.output:  
        print(f'batch_size: {opt.batch_size}')  
        print(f'epochs (niters) : {opt.epochs}')  
        print(f'learning rate : {opt.lr}')  
        print(f'outdir:{opt.out_dir}')
  
    return opt  
  
if __name__ == '__main__':  
    opt = get_options()

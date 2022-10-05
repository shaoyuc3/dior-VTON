"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
"""
import time
from options.train_options import TrainOptions
from datasets import create_dataset, create_visual_ds
from utils.visualizers import define_visualizer
from utils.train_utils import *
from models import create_model
import os, torch

def generate_val_img(visual_ds, model, opt, generate_out_dir, step=0):
    model.eval()
    
    data = visual_ds.get_attr_visual_input()
    
    model.isTrain = False
    # generate
    print("generating for test split")
    with torch.no_grad():
        model.set_input(data)        # unpack data from dataset and apply preprocessing
        model.forward()
        opt.count = model.save_visual(generate_out_dir, opt.count, crop_size=opt.crop_size)
            
    model.train()
        
def main():
    opt = TrainOptions().parse()   # get training options
    opt.crop_size = tuple(map(int, opt.crop_size.split(', ')))
    if opt.square: #opt.crop_size >= 250:
        opt.crop_size = (opt.crop_size[0], opt.crop_size[0])
    opt.square = False
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    visual_ds = create_visual_ds(opt)
    
    # set up model
    model = create_model(opt)      # create a model given opt.model and other options
    load_iter = model.setup(opt)               # regular setup: load and print networks; create schedulers
    if load_iter != -1:
        opt.epoch_count = load_iter
    total_iters = opt.epoch_count
    print("[init] start from iter %d" % total_iters)
    opt.run_test = not opt.no_trial_test
    opt.count = 1
    
    generate_out_dir = os.path.join(opt.val_output_dir + "_%s"%opt.epoch)
    print("generate images at %s" % generate_out_dir)
    if not os.path.exists(generate_out_dir):
        os.mkdir(generate_out_dir)
        
    progressive_steps = {}
    if opt.progressive:
        progressive_steps = get_progressive_training_policy(opt)
        curr_step = max(0, len([i for i in progressive_steps if i<total_iters]) - 1)
        if curr_step < len(progressive_steps):
            keys = list(progressive_steps.keys())
            bs, cs, coe = progressive_steps[keys[curr_step]]
            print("[progressive] init - iter %d, bs: %d, crop: %d" % (total_iters, bs, cs))
            model, dataset, visual_ds = progressive_adjust(model, opt, bs, cs, coe, square=opt.square)
        
                
    
    # train
    epoch_start_time = time.time()  # timer for entire epoch
    while total_iters < opt.n_epochs + opt.n_epochs_decay + 1: 
        for data in dataset:  # inner loop within one epoch
            total_iters += 1
            if total_iters > opt.n_epochs + opt.n_epochs_decay + 1:
                break
                
            
            # model update
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            
            # progressive adjust
            if opt.progressive and (total_iters - 1) in progressive_steps:
                bs, cs, coe = progressive_steps[total_iters - 1]
                print("at total_iter %d, bs: %d, crop: %d" % (total_iters, bs, cs))
                model, dataset, visual_ds = progressive_adjust(model, opt, bs, cs, coe, square=opt.square)
                break
                
            # log
            if total_iters % opt.print_freq == 0:   
                losses = model.get_cum_losses()
                #t_comp = (time.time() - epoch_start_time) / opt.batch_size
                out_string = "[%s][iter-%d]" % (opt.name,  total_iters)
                for loss_name in losses:
                    out_string += "%s: %.4f, " % (loss_name, losses[loss_name])
                print(out_string)
                
           
            # save latest ckpt  
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (total_iters %d)' % (total_iters))
                model.save_networks('latest', total_iters)
                print('End of iter %d / %d \t Time Taken: %d sec' % (total_iters, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            
            # save periodic ckpt
            if total_iters % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of iters %d' % (total_iters))
                save_suffix = 'iter_%d' % total_iters 
                model.save_networks(save_suffix)
            
            # tensorboard
            if total_iters % opt.display_freq == 0:   #
                #model.compute_visuals(total_iters, loss_only=False)
                generate_val_img(visual_ds, model, opt, generate_out_dir, step=total_iters)
                print("at %d, compute visuals" % total_iters)
                
            # update learning rate
            if total_iters % opt.lr_update_unit == 0:
                print(total_iters)
                model.update_learning_rate()                     # update learning rates at the end of every iteration.
                

    model.save_networks('latest', total_iters)


if __name__ == "__main__":
    main()
            

        
           
            

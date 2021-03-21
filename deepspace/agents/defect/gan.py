
import shutil
from tqdm import tqdm

import torch
from torch.optim import lr_scheduler
from torch import nn

from deepspace.agents.defect.base import BaseAgent
from deepspace.graphs.models.gan.dcgan import Generator, Discriminator
from deepspace.datasets.defect.defect import DefectDataLoader
from deepspace.graphs.weights_initializer import xavier_weights
from deepspace.utils.metrics import AverageMeter, AverageMeterList

from deepspace.config.config import config, logger


class GanAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        # define models
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # Create instance from the optimizer
        self.optimizer_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.settings.learning_rate if 'learning_rate' in config.settings else 1e-3,
            betas=config.settings.betas if 'betas' in config.settings else [0.9, 0.999],
            eps=config.settings.eps if 'eps' in config.settings else 1e-8,
            weight_decay=config.settings.weight_decay if 'weight_decay' in config.settings else 1e-5)
        self.optimizer_dis = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.settings.learning_rate if 'learning_rate' in config.settings else 1e-3,
            betas=config.settings.betas if 'betas' in config.settings else [0.9, 0.999],
            eps=config.settings.eps if 'eps' in config.settings else 1e-8,
            weight_decay=config.settings.weight_decay if 'weight_decay' in config.settings else 1e-5)

        # Define Scheduler
        def lambda1(epoch): return pow(1 - epoch / config.settings.max_epoch, 0.9)
        self.scheduler_gen = lr_scheduler.LambdaLR(self.optimizer_gen, lr_lambda=lambda1)
        self.scheduler_dis = lr_scheduler.LambdaLR(self.optimizer_dis, lr_lambda=lambda1)

        # self.generator.apply(xavier_weights)
        # self.discriminator.apply(xavier_weights)

        # define data_loader
        self.data_loader = DefectDataLoader()

        # define loss
        self.loss = nn.BCELoss()
        self.loss = self.loss.to(self.device)
        self.image_loss = nn.MSELoss()
        self.image_loss = self.image_loss.to(self.device)

        # # define metrics
        # self.metrics = nn.BCELoss()
        # self.metrics = self.metrics.to(self.device)
        self.gen_best_metric = 100.0
        self.dis_best_metric = 100.0

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint()

        with torch.no_grad():
            self.real_labels = torch.FloatTensor(config.settings.batch_size).fill_(1).to(self.device)
            self.fake_labels = torch.FloatTensor(config.settings.batch_size).fill_(0).to(self.device)

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in tqdm(range(self.current_epoch, config.settings.max_epoch), desc="traing at -{}-, total epoches -{}-".format(self.current_epoch, config.settings.max_epoch)):
            self.current_epoch = epoch
            gen_loss, dis_loss = self.train_one_epoch()
            # ssim_score = self.validate()
            self.scheduler_gen.step()
            self.scheduler_dis.step()
            # self.scheduler.step(np.round_(ssim_score, 4))
            is_best = gen_loss < self.gen_best_metric or dis_loss < self.dis_best_metric
            if gen_loss < self.gen_best_metric:
                self.gen_best_metric = gen_loss
            if dis_loss < self.dis_best_metric:
                self.dis_best_metric = dis_loss
            self.save_checkpoint(config.settings.checkpoint_file,  is_best=is_best)

    def train_one_epoch(self):
        """
            One epoch of training
            :return:
            """
        # Initialize tqdm dataset
        tqdm_batch = tqdm(
            self.data_loader.train_loader,
            total=self.data_loader.train_iterations,
            desc="Epoch-{}-".format(self.current_epoch))
        # Set the model to be in training mode (for batchnorm)
        self.generator.train()
        self.discriminator.train()
        # Initialize average meters
        dis_epoch_loss = AverageMeter()
        dis_epoch_fake_loss = AverageMeter()
        dis_epoch_normal_loss = AverageMeter()
        gen_epoch_loss = AverageMeter()
        gen_epoch_dis_loss = AverageMeter()
        gen_epoch_image_loss = AverageMeter()
        # loop images
        for defect_images, normal_images in tqdm_batch:
            defect_images = defect_images.to(self.device, dtype=torch.float32)
            normal_images = normal_images.to(self.device, dtype=torch.float32)
            defect_images.requires_grad = True
            normal_images.requires_grad = True

            # start the discriminator by training with real data---
            self.optimizer_dis.zero_grad()
            out_labels = self.discriminator(normal_images)
            dis_loss_normal = self.loss(out_labels.squeeze(), self.real_labels[0:defect_images.shape[0]])
            dis_loss_normal.backward()

            # train disctriminator with defect data may result on difficulty of loss reduce. so skip this step
            # out_labels = self.discriminator(defect_images)
            # dis_loss_defect = self.loss(out_labels.squeeze(), self.fake_labels[0:defect_images.shape[0]])
            # dis_loss_defect.backward()

            # # train the discriminator with fake data---
            fake_images = self.generator(defect_images)
            out_labels = self.discriminator(fake_images.detach())
            dis_loss_fake = self.loss(out_labels.squeeze(), self.fake_labels[0:defect_images.shape[0]])
            dis_loss_fake.backward()

            self.optimizer_dis.step()
            dis_epoch_normal_loss.update(dis_loss_normal.item())
            dis_epoch_fake_loss.update(dis_loss_fake.item())
            dis_epoch_loss.update(dis_loss_normal.item())
            # dis_epoch_loss.update(dis_loss_defect.item())
            dis_epoch_loss.update(dis_loss_fake.item())

            # train the generator now---
            self.optimizer_gen.zero_grad()
            out_labels = self.discriminator(fake_images)
            gen_diss_loss = self.loss(out_labels.squeeze(), self.real_labels[0:defect_images.shape[0]])    # fake labels are real for generator cost
            image_loss = self.image_loss(fake_images, normal_images)
            gen_loss = (1 - config.settings.image_loss_weight) * gen_diss_loss + config.settings.image_loss_weight * image_loss
            gen_loss.backward()

            self.optimizer_gen.step()
            gen_epoch_dis_loss.update(gen_diss_loss.item())
            gen_epoch_image_loss.update(image_loss.item())
            gen_epoch_loss.update(gen_loss.item())

            self.current_iteration += 1

        # close dataloader
        tqdm_batch.close()
        # logging
        self.summary_writer.add_scalar("epoch-training/gen_loss", gen_epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch-training/gen_diss_loss", gen_epoch_dis_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch-training/gen_image_loss", gen_epoch_image_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/dis_loss", dis_epoch_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/dis_normal_loss", dis_epoch_normal_loss.val, self.current_iteration)
        self.summary_writer.add_scalar("epoch_training/dis_fake_loss", dis_epoch_fake_loss.val, self.current_iteration)
        logger.info("Training Results at epoch-" + str(self.current_epoch) + " | " + "gen_loss: " + str(gen_epoch_loss.val) + "- dis_loss: " + str(dis_epoch_loss.val)
                    + "gen_diss_loss: " + str(gen_epoch_dis_loss.val) + "gen_image_loss: " + str(gen_epoch_image_loss.val)
                    + "- dis_normal_loss: " + str(dis_epoch_normal_loss.val) + "- dis_fake_loss: " + str(dis_epoch_fake_loss.val))
        # save images
        # save the reconstructed image
        fake_images = fake_images.squeeze().detach().cpu().numpy()
        defect_images = defect_images.squeeze().detach().cpu().numpy()
        normal_images = normal_images.squeeze().detach().cpu().numpy()
        self.save_output_images(fake_images, defect_images, normal_images)
        return gen_epoch_loss.val, dis_epoch_loss.val

    def test(self):
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations, desc="test at -{}-".format(self.current_epoch))
        self.generator.eval()
        self.discriminator.eval()
        dis_epoch_loss = AverageMeter()
        dis_epoch_fake_loss = AverageMeter()
        dis_epoch_normal_loss = AverageMeter()
        gen_epoch_loss = AverageMeter()
        gen_epoch_dis_loss = AverageMeter()
        gen_epoch_image_loss = AverageMeter()
        with torch.no_grad():
            for defect_images, normal_images, ground_truth_images, paths in tqdm_batch:
                defect_images = defect_images.to(self.device, dtype=torch.float32)
                normal_images = normal_images.to(self.device, dtype=torch.float32)
                ground_truth_images = ground_truth_images.to(self.device, dtype=torch.float32)
                # test discriminator
                fake_images = self.generator(defect_images,)
                fake_labels = self.discriminator(fake_images)
                dis_fake_loss = self.loss(fake_labels, self.fake_labels)
                normal_labels = self.discriminator(normal_images)
                dis_normal_loss = self.loss(normal_labels, self.real_labels)
                dis_loss = dis_fake_loss + dis_normal_loss
                dis_epoch_fake_loss.update(dis_fake_loss.item())
                dis_epoch_normal_loss.update(dis_normal_loss.item())
                dis_epoch_loss.update(dis_loss.item())
                # test generator
                gen_dis_loss = self.loss(fake_labels, self.fake_labels)
                gen_image_loss = self.image_loss(fake_images, normal_images)
                gen_epoch_dis_loss.update(gen_dis_loss.item())
                gen_epoch_image_loss.update(gen_image_loss)
                gen_loss = gen_dis_loss + gen_image_loss
                gen_epoch_loss.update(gen_loss.item())

                # save the reconstructed image
                fake_images = fake_images.squeeze().detach().cpu().numpy()
                defect_images = defect_images.squeeze().detach().cpu().numpy()
                normal_images = normal_images.squeeze().detach().cpu().numpy()
                ground_truth_images = ground_truth_images.squeeze().detach().cpu().numpy()
                self.save_output_images(fake_images, defect_images, normal_images, ground_truth_images, paths)
            # logging
            logger.info("test Results at epoch-" + str(self.current_epoch) + " | " + ' dis_epoch_loss: ' + str(dis_epoch_loss.val) + " | " + ' gen_epoch_loss: ' + str(gen_epoch_loss.val)
                        + " | " + ' dis_epoch_fake_loss: ' + str(dis_epoch_fake_loss.val) + " | " + ' dis_epoch_normal_loss: ' + str(dis_epoch_normal_loss.val)
                        + " | " + ' gen_epoch_dis_loss: ' + str(gen_epoch_dis_loss.val) + " | " + ' gen_epoch_image_loss: ' + str(gen_epoch_image_loss.val))
            tqdm_batch.close()

    def load_checkpoint(self):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        file_name = config.swap.checkpoint_dir / config.settings.checkpoint_file
        try:
            logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.discriminator.load_state_dict(checkpoint['dis_state_dict'])
            self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
            self.optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
            logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                        .format(config.swap.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(config.swap.checkpoint_dir))
            logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'gen_state_dict': self.generator.state_dict(),
            'dis_state_dict': self.discriminator.state_dict(),
            'optimizer_gen': self.optimizer_gen.state_dict(),
            'optimizer_dis': self.optimizer_dis.state_dict(),
        }
        # Save the state
        torch.save(state, config.swap.checkpoint_dir / file_name)
        # backup model on a certain steps
        if self.current_epoch % config.settings.save_model_step == 0:
            torch.save(state, config.swap.checkpoint_dir / (str(self.current_epoch) + '_' + config.settings.checkpoint_file))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(
                config.swap.checkpoint_dir / file_name,
                config.swap.checkpoint_dir / 'model_best.pth.tar')

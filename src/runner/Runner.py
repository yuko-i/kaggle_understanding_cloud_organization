import torch
import time
from util.util import AverageMeter

class Runner():
    def __init__(self):
        return

    def train(self,
              model,
              criterion,
              optimizer,
              loaders,
              logger,
              metrics,
              scheduler,
              device: str='cpu',
              epoch_num: int=16,
              log_dir: str='',
              ):

        logger.info('========= Train =========')
        iter_num = int(len(loaders["train"].dataset) / loaders["train"].batch_size)

        for epoch in range(0, epoch_num):
            # ============= #
            # Train
            # ============= #
            model.train(True)
            start = time.time()
            train_loss = AverageMeter()#[]
            valid_loss = AverageMeter()#[]
            dice_cof = AverageMeter()#[]

            for batch_idx, (data, target) in enumerate(loaders['train']):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                dice = metrics(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.update(loss.item() * data.size(0))
                dice_cof.update(dice.item() * data.size(0))
                #train_loss.append(loss.item() * data.size(0))
                #dice_cof.append(dice.item() * data.size(0))

                # Log
                if batch_idx % 100 == 0:
                    logger.info(f'{epoch} epoch | '
                                f'{batch_idx}/{iter_num} | '
                                f'loss: {train_loss.avg:.3f} | '#f'loss: {train_loss[-1]:.3f} | '
                                f'dice: {dice_cof.avg:.3f}')#f'dice: {dice_cof[-1]:.3f}')
            # ============= #
            # valid
            # ============= #
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(loaders['valid']):
                    data, target = data.to(device), target.to(device)

                    output = model(data)
                    loss = criterion(output, target)
                    valid_loss.update(loss.item() * data.size(0))
                    #valid_loss.append(loss.item() * data.size(0))

            #train_loss_avg = sum(train_loss) / len(train_loss)
            #valid_loss_avg = sum(valid_loss) / len(valid_loss)
            #dice_cof_avg = sum(dice_cof) / len(dice_cof)
            process_time = time.time() - start

            logger.info(f'{epoch} epoch | '
                        f'train_loss: {train_loss.avg:.3f} | '
                        f'valid_loss: {valid_loss.avg:.3f} | '
                        f'dice_cof: {dice_cof.avg:.3f} | '
                        f'lr: {optimizer.param_groups[0]["lr"]} | '
                        f'time: {process_time}')

            scheduler.step(metrics=valid_loss.avg)
            torch.save(model.state_dict(), f'{log_dir}/train_model_{epoch}.pth')




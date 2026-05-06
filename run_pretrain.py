from pretrain_custom_lm import TrainLMConfig, train_lm, save_lm

cfg = TrainLMConfig()
cfg.num_epochs = 1
cfg.batch_size = 8
cfg.max_seq_len = 128
cfg.save_dir = 'checkpoints_lm'

model = train_lm(cfg)
save_lm(model, 'epistemic_lm.pt')
print('Done pretraining')

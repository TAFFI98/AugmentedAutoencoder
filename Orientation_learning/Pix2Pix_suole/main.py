import tensorflow as tf
import os
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
from model import Generator,Discriminator,generator_loss,discriminator_loss
from utils import load_image_train,load,load_image_test,generate_images


def train(PATH,CHECKPOINT_PATH,LOG_PATH,restore_checkpoint):
 # The training set consist of 76 images
 BUFFER_SIZE = 76
 # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
 BATCH_SIZE = 5
 # Each image is resized 256x256

 OUTPUT_CHANNELS = 3


 train_dataset = tf.data.Dataset.list_files(str(PATH + 'train/*.png'))
 train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
 train_dataset = train_dataset.shuffle(BUFFER_SIZE)
 train_dataset = train_dataset.batch(BATCH_SIZE)
 #print(train_dataset.take(1))

 test_dataset = tf.data.Dataset.list_files(str(PATH + 'test/*.png'))
 test_dataset = test_dataset.map(load_image_test)
 test_dataset = test_dataset.batch(BATCH_SIZE)

 #print(test_dataset.take(1))
 generator = Generator(OUTPUT_CHANNELS=OUTPUT_CHANNELS)
 generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

 discriminator = Discriminator()
 discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

 #Plot the Generator and discriminator models
 #tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
 #tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

 def discriminator_loss(disc_real_output, disc_generated_output):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

 def generator_loss(disc_generated_output, gen_output, target):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  LAMBDA = 100
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

 checkpoint_dir = CHECKPOINT_PATH
 checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
 checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)


 log_dir = LOG_PATH

 summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

 @tf.function
 def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
   gen_output = generator(input_image, training=True)

   disc_real_output = discriminator([input_image, target], training=True)
   disc_generated_output = discriminator([input_image, gen_output], training=True)

   gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
   disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
   tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
   tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
   tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
   tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)

 def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
   if (step) % 10000 == 0:
    display.clear_output(wait=True)

    if step != 0:
     print(f'Time taken for 10000 steps: {time.time() - start:.2f} sec\n')

    start = time.time()

    generate_images(generator, example_input, example_target)
    print(f"Step: {step // 1000}k")

   train_step(input_image, target, step)

   # Training step
   if (step + 1) % 10 == 0:
    print('.', end='', flush=True)

   # Save (checkpoint) the model every X steps
   if (step + 1) % 20000 == 0:
    checkpoint.save(file_prefix=checkpoint_prefix)
 # Restoring the latest checkpoint in checkpoint_dir
 if restore_checkpoint == True:
  checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH))
  
 fit(train_dataset, test_dataset, steps=60000)


def test(PATH,CHECKPOINT_PATH):
 BUFFER_SIZE = 20
 # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
 BATCH_SIZE = 1
 # Each image is resized 256x256
 IMG_WIDTH = 256
 IMG_HEIGHT = 256
 OUTPUT_CHANNELS = 3
 test_dataset = tf.data.Dataset.list_files(str(PATH + 'test/*.png'))
 test_dataset = test_dataset.map(load_image_test)
 test_dataset = test_dataset.batch(BATCH_SIZE)

 generator = Generator(OUTPUT_CHANNELS=OUTPUT_CHANNELS)
 discriminator = Discriminator()

 generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
 discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
 checkpoint_dir = CHECKPOINT_PATH
 checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)
 # Restoring the latest checkpoint in checkpoint_dir
 checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
 # Run the trained model on a few examples from the test set
 for inp, tar in test_dataset.take(20):
  generate_images(generator, inp, tar)


if __name__ == "__main__":
 #Define the path to the  dataset - inside thia folder there iare 2 sub folers, one used for trainig the other for validation
 PATH='/home/user/Orientation_learning/Pix2Pix_suole/dataset_augmented_2/'
 #Define the path where the training checkpoints are going to be saved
 CHECKPOINT_PATH = "/home/user/Orientation_learning/Pix2Pix_suole/checkpoints model_2/"
 #Define the path where the training logs are going to be saved
 LOG_PATH = "/home/user/Orientation_learning/Pix2Pix_suole/logs/"


 # TRAIN THE MODEL

 #train(PATH, CHECKPOINT_PATH, LOG_PATH, False)


 # TEST THE MODEL

 test(PATH, CHECKPOINT_PATH)
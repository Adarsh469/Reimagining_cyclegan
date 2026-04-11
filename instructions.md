## Reimagine CycleGans

### Instructions on adding loss functions:

Work on your loss functions by making a copy of original cycle_gan_model under /models. 

Discriminators:
1. function backward_D_basic is used to calculate the loss, make a copy of this function
2. Edit this function to suite the experiment
3. The loss is finally calculated for each discriminator in backward_D_B() and backward_D_B(), I personally like to add conditional statements here so that i can use different loss functions by passing a parameter.
4. optionally add a parameter in the constructor of the model to select the loss function, let to fall to default if noting specified.

Generator:
1. The loss classes for the generator are initialised in the constructor.
2. in the constructor change the loss_class to whatever the loss you are working with
3. to make it more flexible, i advice adding of conditional statements here also




~ LaserHammer




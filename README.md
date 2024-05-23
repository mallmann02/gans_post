<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>

# GAN'S

Before starting, let's first introduce what is our study object this time. So, GAN's are an acronym for Generative Adversarial Networks, let's take a brief look of each of these words mean.

GAN's are a class of `G`enerative models that are trained through an `A`dversarial min-max game between two neural `N`etworks, the generator and the discriminator. We'll talk about each of the magical words here, but first things first, let's see where we are in the area of ML and statistics.

### Looking at the big picture, what are generative models? And how are they different from our old friend class of dicriminative models?

In a intuitive (super) high-level, generative models are a class of models that are used to `generate` new data samples from a given distribution. In the other hand, discriminative models are used to `discriminate` between different kinds of data samples. Enough of the words playing, let's talk under a statistical perspective.

![Discriminative vs Generative Models](./images/generative_v_discriminative.png "Discriminative vs Generative Models")

Source: [Google Developers](https://developers.google.com/machine-learning/gan/generative)

The generative models are used to model the joint distribution of the input data and the labels, $P(X, Y)$ while the discriminative models are used to model the conditional distribution of the labels given the input data, $P(Y given X)$.

Ok, but let's have more intuition about it. For this let's review the concept of <u>conditional probability</u> and <u>joint probability</u>. The joint probability is the probability of two events happening at the same time, while the conditional probability is the probability of one event happening given that another event has already happened.

Looking at both formulas, interstingly, we have some kind of relationship between them (whispers: the Bayes' Theorem):

$$ P(X, Y) = P(X) \cdot P(Y|X) $$

$$ P(Y|X) = \frac{P(X, Y)}{P(X)} $$

Bringing to the example of digits recognition, the discriminative model, in the training step, has access to the input data $X$ that are the pixels $x_1, x_2, ..., x_n$ of the image and the labels $Y$ that are the digits $0, 1, ..., 9$. So, thinking in a simple classification network, it works by <u>maximizing the likelihood</u> of the labels given the input data, $P(YgivenX)$, and it does this by receiving a feedback in the form of "it's right" or "it's wrong" and adjusting the weights in favor of maximizing the likelihood for the specific class $Y$.

In the other hand, the generative model, is all blindfolded and has no idea of the labels nor the input data. Instead, it receives a random input data in the form of a noise vector $z$ and gradually learns to transform it in the "real" input data distribution by maximizing the joint probability: $P(x_1), P(x_2), ..., P(x_n)$, that is, the probability of the pixels of the image occuring together in a certain way. And it does this by receiving a feedback that is the <u>distance</u> of its "transformation" of the data from a real data sample.

### What about the adversarial part of these models?

Now that we have brief over the generative and discriminative models, let's talk about the interaction between them and how we take advantage of this to "generate" new things.

So the adversial part of the GAN's comes from the fact that, although we have two separate neural networks, the optimization process for both are entangled. The advantage of this process establishes as the competition between them leads to an equilibrium of the generator as it, during the training, is using the feedback from the discriminator to drive the approximattion of the sample generation towards the real data distribution.

So, let's get a intuition on this parting from a kind example. Imagine that are sampling random numbers from a certain distribution, let's say for example a Normal Distribution, so we take about 100 samples and plot them in a histogram. This will be our real data distribution. We want to figure out which distribution is this and we will use a feedback loop for this, pretty much like a game of Marco Polo. Let's see the steps:

1. We don't have idea of the distribution, so we start, like in Marco Polo, by saying "Marco": sampling random numbers.
2. The other player will gives us a feedback, saying "Polo", so we know how far we are from our objective.
3. Back in our turn, we take a step in the direction of the feedback and keep repeating the process until we get near the other player.

To visualize it in the example of the distribution, take a look at the GIF below:

<video width="100%" height="240" controls>
  <source src="./images/convergence_ex.mov" type="video/mp4">
</video>

In the GIF above, we have a simple example on two Normal Distributions. The blue one is our generated distribution that gradually converges to the real data distribution as it changes it's parameters of mean and variance.

For a more precise example of visualizing the training process of a GAN, take a look at this [GAN Lab](https://poloclub.github.io/ganlab/). There you can draw your own data distribution and see the generator trying to approximate it while the discriminator learns the function that separates the real data from the generated data.

## Architecture visualization and the training process

![Image](./images/gan_architecture.jpeg "GAN Architecture")

Source: [AWS](https://aws.amazon.com/what-is/gan)

Is the image above we can see the <u>main components</u> for the GAN architecture.

### Generator

Following our image above, we have our Generator being the yellow box. By it's side, we have little purple triangle that is inputing something in our box, and this the "random noise" - or more simply random numbers following a certain distribution - that we talked about in the beginning. Inside the box, we are doing a transformation of this random noise in a certain form to get a more positive feedback from the discriminator.

Just to not turn our yellow box into a black box, let's talk about the transformation that we are doing. Do you remember when we talked about the generative model and the joint probability? So, the generator is trying to learn the joint probability of the input data $P(X)$, and this learning process is done by a neural network that is trained through the feedback of the discriminator about how distant the generated data is from the real data distribution.

### Discriminator

Also following our image above, we have our Discriminator being the blue box that is receiving two inputs, the "Real Sample" and the "Fake Sample". Inside the box, you can think what's happening as a simple classification process, where a neural network outputs how likely the "Fake Sample" was sampled from the real data distribution. As we'll see, this likelihood is defined by a distance measure that can come is different flavors.

## Loss Functions

### Discriminator Loss

The discriminator loss is a measure of how well the discriminator is able to distinguish between real and fake samples. The discriminator loss is calculated by taking the difference between the output of the discriminator for the real and fake samples. In the original GAN paper, the discriminator loss is defined as the cross-entropy loss between the real and fake samples.

The loss is defined in the following formula:

$$ L_D = -{E}_{x \sim p_{data}(x)}[\log D(x)] - {E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))] $$

Where:
- $L_D$ is the discriminator loss
- $x$ is a real sample from the dataset
- $p_{data}(x)$ is the distribution of the real samples
- $D(x)$ is the output of the discriminator for the real sample $x$
- $z$ is a random noise vector
- $G(z)$ is the output of the generator for the random noise vector $z$
- $p_{z}(z)$ is the distribution of the random noise vectors

### Generator Loss

The generator loss is a measure of how well the generator is able to generate samples that fool the discriminator. The generator loss is calculated by taking the negative of the output of the discriminator for the fake samples.

The generator loss is calculated using the following formula:

$$ L_G = -{E}_{z \sim p_{z}(z)}[\log D(G(z))] $$

Where:
- $L_G$ is the generator loss
- $z$ is a random noise vector
- $G(z)$ is the output of the generator for the random noise vector $z$
- $D(G(z))$ is the output of the discriminator for the fake sample $G(z)$
- $p_{z}(z)$ is the distribution of the random noise vectors

## Training Steps

So, now we have a little intuition about the architecture and how our components conects with each other through the loss functions. Let's talk about the training steps of the GAN's.

1. sample a batch of random noise vectors from a given distribution.

2. sample a batch of real samples from the dataset.

3. feed the random noise vectors into the generator to generate a batch of fake samples.

4. feed the real samples and the fake samples into the discriminator to get the discriminator loss.

5. update the discriminator weights based on the gradients given by the discriminator loss.

6. sample a new batch of random noise vectors from the distribution.

7. feed the random noise vectors into the generator to generate a new batch of fake samples.

8. feed the fake samples into the discriminator to get the generator loss.

9. update the generator weights using the gradients of the generator loss.

## Code analysis

Following the above training steps let's look up to a code of a [DC-GAN using PyTorch](https://neptune.ai/blog/gan-failure-modes) to generate images of the MNIST dataset. We'll analyze the code in detail to understand where each step fits and highlight some tricks that are used to stabilize the training process.

### Training loop

Let's take a look at the main training loop just to have a sense of the steps before entering inside the architectures:

```python
# Iterating over the epochs
for epoch in range(epochs):
  loss_g = 0.0
  loss_d_real = 0.0
  loss_d_fake = 0.0

  # Iterating over the real data batches
  for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size)):
    # Getting the real data from the batch (Step 2)
    image, _ = data

    # Moving the data to the device (Pytorch)
    image = image.to(device)

    # Getting the batch size
    b_size = len(image)

    # [Trick] Training the discriminator for consecutive steps
    for step in range(k):
      # Creating the noise vector (Step 1)
      # Getting the fake data from the generator (Step 3)
      data_fake = generator(create_noise(b_size, latent_dim)).detach()

      data_real = image

      # Training the discriminator (Step 4 and 5)
      loss_d_fake_real = train_discriminator(optim_d, data_real, data_fake)
    
    # Creating aonther noise vector (Step 6)
    # Getting the fake data from the generator (Step 7)
    data_fake = generator(create_noise(b_size, latent_dim))

    # Training the generator (Step 8 and 9)
    loss_g += train_generator(optim_g, data_fake)
```

### [Looking inside] `train_generator` function

```python
def train_generator(optimizer, data_fake):
  # Getting the batch size
  b_size = data_fake.size(0)

  # Creating the a vector of ones to sinalize the real data
  real_label = label_real(b_size)

  # Zeroing the gradients
  optimizer.zero_grad()

  # Getting output from the discriminator for the generated data
  output = discriminator(data_fake)

  # Calculating the loss
  # Here we are seeing how well the generator is able to fool the discriminator by seeing how likely the discriminator is to classify the fake data as real
  loss = criterion(output, real_label)

  # Backpropagating the loss
  loss.backward()
  optimizer.step()
  return loss
```

### [Looking inside] `train_discriminator` function

```python
def train_discriminator(optimizer, data_real, data_fake):
  # Getting the batch size
  b_size = data_real.size(0)

  # Creating the a vector of ones to sinalize the real data
  real_label = label_real(b_size)

  # Creating the a vector of zeros to sinalize the fake data
  fake_label = label_fake(b_size)

  # Zeroing the gradients
  optimizer.zero_grad()

  # Getting output from the discriminator for the real data
  output_real = discriminator(data_real)

  # Calculating the loss for the real data
  # [Trick] Here we are seeing how well the discriminator is able to classify the real data as real
  loss_real = criterion(output_real, real_label)

  # Getting output from the discriminator for the fake data
  output_fake = discriminator(data_fake)

  # Calculating the loss for the fake data
  # [Trick] Here we are seeing how well the discriminator is able to classify the fake data as fake
  loss_fake = criterion(output_fake, fake_label)

  # Combining the losses
  # [Trick] Here we are accumulating the losses for the real and fake data to get the total loss for the discriminator
  loss_real.backward()
  loss_fake.backward()

  # Backpropagating the loss
  optimizer.step()
  return loss_real, loss_fake
```

In the code I've marked some tricks that are used to stabilize the training process. Usually, the training process of GAN's is very unstable and it's common to observe some loss values spiking a lot. 

One of the tricks used in the code is the training of the discriminator for k consecutive steps before training the generator. This trick is used to stabilize the training process by giving the discriminator more time to learn the distribution of the real data before the generator starts to generate fake data. By doing this we are able to retrieve a better feedback from the discriminator. Imagine that the discriminator is setting a target for the generator to aim, so it's better if we keep this "target" stable.

Another trick used in the code is train the discriminator for the real and fake data separately and then accumulate the losses. This trick is also used to stabilize the training process by providing a better sense of distance between the real and fake data for the discriminator. In the conventional loss function, we are just calculate the distance between the real and fake data.


## References
- `FANTASTIC PAGE` [GAN Lab](https://poloclub.github.io/ganlab/)
- `Video` [Generative vs. Discriminative AI](https://www.youtube.com/watch?v=hjsZSmL67Ck)
- `Video` [Difference between generative and discriminative models.](https://www.youtube.com/watch?v=cmYQNhv5xUw)
- `Article` [Background: What is a Generative Model?](https://developers.google.com/machine-learning/gan/generative)
- `Article` [Understanding GAN Loss Functions](https://neptune.ai/blog/gan-loss-functions)
- `Article` [GANs Failure Modes: How to Identify and Monitor Them](https://neptune.ai/blog/gan-failure-modes)
- `Article` [GAN Hacks](https://github.com/soumith/ganhacks)
- `Article` [GAN Loss Functions](https://developers.google.com/machine-learning/gan/loss)
- `Article` [GANs: Generative Adversarial Networks](https://aws.amazon.com/what-is/gan)
- `Article` [Understanding Generative Adversarial Networks](https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/)
- `Article` [Joint Probability Distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution)
- `Article` [Conditional Probability Distribution](https://en.wikipedia.org/wiki/Conditional_probability_distribution)
- `Video` [Basic probability: Joint, marginal and conditional probability Independence](https://www.youtube.com/watch?v=SrEmzdOT65s)
- `Paper` [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498)

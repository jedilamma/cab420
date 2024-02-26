import numpy as np
import matplotlib.pyplot as plt




#---------------------------------------------------------------------------------------------------------
# scalar
scalar = np.array(0)
# vector of increasing values 
vector = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9 , 10])
# can create the same vector with vector 
vector = np.arange(0, 11) # note goes to 11 as last one not included 
# creating a diagonal matrix
matrix = np.eye(4)
# 3-D array of random numbers drawn from standard normal distribution
# also demos how to draw random numbers
# will make it of shape [2 x 3 x 4]
three_d_array = np.random.randn(2,3,4)

# lets print them now to have a look
print(f'scalar = {scalar}')
print(f'scalar shape = {scalar.shape}')
print(f'vector = {vector}')
print(f'vector shape = {vector.shape}')
print(f'matrix =\n {matrix}')
print(f'matrix shape = {matrix.shape}')
print(f'3d_array =\n {three_d_array}')
print(f'3d_array shape = {three_d_array.shape}')

# lets extract the first three digits from our vector
vector[0:3]

# lets try the last three digits of the same vector
vector[-3:-1]

# using double colon will say just keep going along this dimension until we get to the end
vector[-3::] 

vector = np.arange(0, 9)
# reshape to a 3 x 3 matrix
matrix = vector.reshape(3, 3)
print(matrix)

vector = np.arange(0, 9)
# reshape to a 3 x 3 matrix
matrix = vector.reshape(3, 3)
print(matrix)

# index first row (zero index in python) and keep every column entry from there
row_1 = matrix[0, :]
print(row_1)

# creating array for x variable
# want say 1000 evenly spaced samples from [0, 2pi]
num_samples = 1000
x = np.linspace(0, 2 * np.pi, num_samples)
y = np.sin(x)
# can print these if you want to look at what is inside yourself
#print(x)
#print(y)


# lets plot the previous image
fig, ax = plt.subplots()  # Create a figure containing a single axes.
# Plot some data on the axes and give it a label so we can add a legend later
ax.plot(x, y, label='sine function')  
# lets add a title and some axis labels
ax.set_xlabel('x variable')
ax.set_ylabel('y variable')
ax.set_title('Sine function')
ax.legend()


#---------------------------------------------------------------------------------------------------------
# lets plot a Gaussian distribution probability density function and a histogram
# of samples from the distribution on the same plot

# creating gaussian PDF
x = np.linspace(-4, 4, 1000)
gaussian_pdf = 1 / (np.sqrt(2 * np.pi)) * np.exp(- 0.5 * x ** 2.0)
# now lets draw a bunch of random samples from a standard normal
# which we can use to create a histogram of
# Will draw 10000 samples 
gaussian_samples = np.random.randn(10000) 
# now lets plot it all
# copying some of the code above and changing just what we need
fig, ax = plt.subplots()  # Create a figure containing a single axes.
# Plot some data on the axes and give it a label so we can add a legend later
# lets also control the color now
# Am using HTML codes for colors here
ax.plot(x, gaussian_pdf, label='True PDF', color='#dd7878', linewidth=5)  
# now lets get the histogram of it
# The histogram should resemble the true probability density function
ax.hist(gaussian_samples, density=True,
        bins=100, label='Estimated PDF',
        color='#209fb5',
        alpha=0.6) # alpha controls the transparency
# lets add a title and some axis labels
ax.set_xlabel('x variable')
ax.set_ylabel('y variable')
ax.set_title('Gaussian PDF')
ax.legend()

#---------------------------------------------------------------------------------------------------------
# importing the cookbook from matplotlib just because it can give us example images
import matplotlib.cbook as cbook
# load in an MRI image from the matplotlib cookbook
with cbook.get_sample_data('s1045.ima.gz') as datafile:
    mri_data = datafile.read()
mri_image = np.frombuffer(mri_data, np.uint16).astype(float).reshape(256, 256)
# lets plot it, but also plot something next to it.
# Lets plot the histogram of pixel intensities next to it
 
# copying some of the code above and changing just what we need
# now need two axes as there is two subplots
# also going to make the figure a little wider
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  
# on the first subplot, plot the image itself
ax1.imshow(mri_image)
ax1.set_title('MRI image')
# now to plot image intensity histograms
# reshape with -1 as the new size will cause it to flatten
# to a vector
ax2.hist(mri_image.reshape(-1), bins=60)
ax2.set_xlabel('Pixel value')
ax2.set_ylabel('Frequency')

#---------------------------------------------------------------------------------------------------------

# get the mnist dataset
import tensorflow as tf
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
# lets create a plot of some of them
# will make a 2 x 8 subplot
# copying some of the code above and changing just what we need
# now need two axes as there is two subplots
# also going to make the figure a little wider
fig, axs = plt.subplots(2, 8, figsize=(16, 6))  
# on the first subplot, plot the image itself
# now we will iterate over few of them.
idx = 0
for col_idx in range(0, 8): 
    for row_idx in range(0, 2): 
        # the np.squeeze will remove and dimensions that have length of one
        # from an array. Say we have an image of shape [1, 28, 28, 1],
        # which can often happen with how our datasets are formed, then the
        # np.squeeze() will make it of shape [28, 28]
        axs[row_idx, col_idx].imshow(np.squeeze(train_x[idx, :, :]))
        axs[row_idx, col_idx].set_title(f'label = {train_y[idx]}')
        # increment the image index counter 
        idx += 1
        
fig.savefig('numbers-nn.svg')
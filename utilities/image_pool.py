import random
import torch


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size, output_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        self.output_size = output_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []
    
    def add(self, images):
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    self.images[random_id] = image
                
            
        return_images = random.sample(self.images, self.output_size)   

    def query(self, output_size=None):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if output_size is None:
            output_size = self.output_size
        return random.sample(self.images, output_size)   
        

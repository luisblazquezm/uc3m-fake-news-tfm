


class ImageProcessor: 
    # Add code for image manipulation detection (Step 2 - Image Processing)

    def __init__(self):
        pass

    def process(self, image_path: str):
        
        try:
            self.__process_image(image_path=image_path)
        except Exception as e:
            return -1

        return 1

    def __process_image(self, image_path: str):
        # Add code for image manipulation detection (Step 2 - Image Processing)
        print("News Image:")
        print(image_path)
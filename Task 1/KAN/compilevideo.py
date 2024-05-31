import cv2
import os

def images_to_video(image_folder, video_name, fps=120):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Ensure the images are in the correct order
    print(images)

    # Read the first image to get the dimensions
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()

    # move the video into the folder and delete all images in the same folder
    os.rename(video_name, os.path.join(image_folder, video_name))
    for image in images:
        os.remove(os.path.join(image_folder, image))
    
    print(f'Video saved as {os.path.join(image_folder, video_name)}')


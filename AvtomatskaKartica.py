import os
import matplotlib.pylab as plt
import cvzone
import time
import random
import smtplib
import ssl
import json
from email.message import EmailMessage
import imghdr
from MediaPipe import *
from StyleTransfer_TensorFlow import *

# Get email config from json file
try:
    with open("Email_config.json", "r") as read_file:
        email_data = json.load(read_file)
        sender_email = email_data["sender_email"]
        print(f"Sender email: {sender_email}")
        sender_email_app_password = email_data["sender_app_password"]
        print(f"Sender password: {sender_email_app_password}")
        receiver_email = email_data["receiver_email"]
        print(f"Receiver email: {receiver_email}")
except Exception as e:
    print(f"Error getting email config data from file: {e}")


def import_slika_osebe():
    # Import prve datoteke v mapi slika osebe
    file_name = os.listdir("slika osebe")[0]
    if file_name is None:
        raise ValueError("Ni slike osebe")
    return file_name


def import_razglednica():
    # Import prve datoteke v mapi razglednica
    file_name = os.listdir("razglednica")[0]
    if file_name is None:
        raise ValueError("Ni razglednice")
    return file_name


def overlay_person(image_person, razglednica_file_name):
    # import razglednica
    razglednica = cv2.imread(f"razglednica/{razglednica_file_name}")
    razglednica_rgb = cv2.cvtColor(razglednica, cv2.COLOR_BGR2RGB)

    # resize person image
    image_person = image_resize(image_person, 500)

    # all possible positions
    potencial_positions = [0, 650, 1000, 1000, 1100]
    # pick random position
    pick = random.randint(0, 4)

    # overlay person at position
    position = [potencial_positions[pick], razglednica_rgb.shape[0]-image_person.shape[0]-5]
    overlaid_image = cvzone.overlayPNG(razglednica_rgb, image_person, position)

    # Return image
    return overlaid_image


def image_resize(image, width=None, height=None):
    # initialize the dimensions of the image to be resized and grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    if width is None:
        # calculate the ratio of the height and construct the dimensions
        ratio = height / float(h)
        dim = (int(w * ratio), height)

    else:
        # calculate the ratio of the width and construct the dimensions
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


def send_email(image):
    # Get image from file
    image_data = image.read()
    image_type = imghdr.what(image.name)
    image_name = image.name

    # Create message
    message = EmailMessage()
    message['Subject'] = "Vaš selfie na razglednici"  # Defining email subject
    message['From'] = sender_email  # Defining sender email
    message['To'] = receiver_email  # Defining receiver email
    message.set_content('Pošiljamo vam vaš selfie na razglednici, Lep pozdrav')  # Defining email body
    # Add photo
    message.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
        server.login(sender_email, sender_email_app_password)
        server.send_message(message)


if __name__ == '__main__':
    # Get names of files
    file_oseba = import_slika_osebe()
    file_razglednica = import_razglednica()

    # Get mask of background
    background_mask = remove_background(file_oseba)

    # Transfer image style from razglednica to selfie
    transferred_slika_osebe = transfer_style(file_oseba, file_razglednica)

    # make the transparent background r = 0, g = 0, b = 0, a = 0
    bg_image = np.zeros(transferred_slika_osebe.shape, dtype=np.uint8)

    # combine person and background image using the mask
    final_slika_osebe = np.where(background_mask, transferred_slika_osebe, bg_image)

    # make alpha channel with transparent background
    alpha_slika_osebe = cv2.cvtColor(final_slika_osebe, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(alpha_slika_osebe, 0, 255, cv2.THRESH_BINARY)

    # add alpha channel to image
    r, g, b = cv2.split(final_slika_osebe)
    a = np.ones(r.shape, dtype=np.uint8) * 255
    final_slika_osebe = cv2.merge((r, g, b, alpha))

    # put person on razglednica
    final_slika = overlay_person(final_slika_osebe, file_razglednica)

    # show final image
    plt.figure()
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(final_slika)
    plt.axis('off')

    # save image to "končne slike" folder, comment to disable saving
    name_of_image = f"končne slike/oseba_na_razglednici_{time.strftime('%H.%M.%S', time.localtime())}.png"
    plt.savefig(name_of_image)

    # Send picture over email
    try:
        with open(name_of_image, 'rb') as image_file:
            send_email(image_file)
    except Exception as e:
        print(f"Error when sending email: {e}")

    # Finally show image, this has to be the last line of code, because it wipes the image from program
    plt.show()

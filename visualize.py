import matplotlib.pyplot as plt


def draw_images(images):
    fig = plt.figure(figsize=(17, 7)) # (27, 17))

    for i, img in enumerate(images):
        a = fig.add_subplot(len(images) // 3 + 1, 3, (i + 1)) # 3
        plt.imshow(img)
        a.set_title(str(i) + "th image")

    plt.tight_layout()
    plt.show()

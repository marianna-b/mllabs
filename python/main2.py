import gzip
import pickle
import shutil
from typing import List, Tuple

import numpy as np

from python.network import deserialize_from_file


def load():
    with gzip.open('./data/mnist.pkl.gz', 'rb') as f:
        tr_d, va_d, te_d = pickle.load(f, encoding='latin1')

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = tr_d[1]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data


def save_image_to_file(grayscaled_1d_array: np.ndarray, filepath: str,
                       tuple_with_new_sizes: Tuple[int, int] = (28, 28)):
    from PIL import Image
    image_matrix = np.resize(grayscaled_1d_array, tuple_with_new_sizes)
    image = Image.fromarray((1 - image_matrix) * 255).convert('RGB')
    image.save(filepath)


def save_wrong_results(results: List[Tuple[np.ndarray, int, int]],
                       folder_prefix: str = "./wrong"):
    import os
    import collections
    os.makedirs(folder_prefix, exist_ok=True)
    failed_dict = collections.defaultdict(lambda: 1)
    for x, y, res in results:
        pair = (y, res)
        times = failed_dict[pair]
        failed_dict[pair] += 1
        save_image_to_file(x, "{}/expected-{}-got-{}[{}].jpg".format(folder_prefix, y, res, times))


train_data, valid_data, test_data = load()

# n = Network([784, 30, 10])
# n.fit(train_data, 30, 10, 3.0)
# Network.serialize_to_file(n, "saved_network.pkl.gz")

n = deserialize_from_file('./data/saved_network.pkl.gz')

# print(len(valid_data))
# print(len(test_data))
# print(len(train_data))


amount, wrongs = n.validate(test_data)
print(amount / len(test_data))
# shutil.rmtree("./wrong", ignore_errors=True)
save_wrong_results(wrongs)


# for x, y in valid_data:
#     n.train(x, y, 0.1)
#
# amount, wrongs = n.validate(test_data)
# print(amount, len(test_data))

# todo Button and input for accepting your mouse-written digits!
def tk_loop():
    from PIL import ImageDraw, Image
    from tkinter import Canvas, Tk, YES, BOTH, BOTTOM, mainloop, Label
    from abc import ABCMeta, abstractmethod
    scale = 30
    canvas_width = 28 * scale
    canvas_height = 28 * scale

    class DrawingApi(metaclass=ABCMeta):
        @abstractmethod
        def draw_circle(self, x1, y1, x2, y2, color='Black'):
            pass

        @abstractmethod
        def fill_with_color(self, x1, y1, x2, y2, color='White'):
            pass

        @abstractmethod
        def clear(self):
            pass

    class ImageDrawingApi(DrawingApi):
        def clear(self):
            pass

        def __init__(self, img: Image):
            self.drawing = ImageDraw.Draw(img)

        def draw_circle(self, x1, y1, x2, y2, color='Black'):
            self.drawing.ellipse([x1, y1, x2, y2], fill=color)

        def fill_with_color(self, x1, y1, x2, y2, color='White'):
            self.drawing.rectangle([x1, y1, x2, y2], fill=color)

    class CanvasDrawingApi(DrawingApi):
        def clear(self):
            self.drawing.delete("all")

        def __init__(self, canvas: Canvas):
            self.drawing = canvas

        def draw_circle(self, x1, y1, x2, y2, color='Black'):
            self.drawing.create_oval(x1, y1, x2, y2, fill=color)

        def fill_with_color(self, x1, y1, x2, y2, color='White'):
            self.drawing.create_rectangle(x1, y1, x2, y2, fill=color, width=0)

    class BatchDrawing(DrawingApi):
        def clear(self):
            for x in self.batch:
                x.clear()

        def __init__(self, batch: List[DrawingApi]):
            self.batch = batch

        def draw_circle(self, x1, y1, x2, y2, color='Black'):
            for x in self.batch:
                x.draw_circle(x1, y1, x2, y2, color)

        def fill_with_color(self, x1, y1, x2, y2, color='White'):
            for x in self.batch:
                x.fill_with_color(x1, y1, x2, y2, color)

    DrawingApi.register(ImageDrawingApi)
    DrawingApi.register(CanvasDrawingApi)
    DrawingApi.register(BatchDrawing)

    def paint(event):
        black = "#000000"
        x1, y1 = (event.x - scale), (event.y - scale)
        x2, y2 = (event.x + scale), (event.y + scale)
        batch_drawing.draw_circle(x1, y1, x2, y2, black)
        message['text'] = predict()

    def predict():
        kek = fake_canvas.copy()
        kek.thumbnail((28, 28), Image.ANTIALIAS)
        arr = 1 - np.array(kek).reshape(784, 1) / 255
        return str(n.predict_digit(arr))

    def clear_it(event):
        kek = fake_canvas.copy()
        kek.thumbnail((28, 28), Image.ANTIALIAS)
        kek.save("./saved/[{}]-{}.jpg".format(global_counter[0], message['text']))
        global_counter[0] += 1
        batch_drawing.clear()
        batch_drawing.fill_with_color(0, 0, canvas_width, canvas_height)

    import os
    global_counter = [1]
    shutil.rmtree("./saved/", ignore_errors=True)
    os.makedirs("./saved", exist_ok=True)

    master = Tk()
    master.resizable(0, 0)
    w = Canvas(master,
               width=canvas_width,
               height=canvas_height, bg="white", cursor="circle", bd=4)

    fake_canvas = Image.new("L", (28 * scale, 28 * scale), color=255)
    fake_drawing = ImageDrawingApi(fake_canvas)
    canvas_drawing = CanvasDrawingApi(w)
    w.config(highlightbackground="brown")

    # canvas_drawing.fill_with_color(0, 0, canvas_width, canvas_height)
    batch_drawing = BatchDrawing([fake_drawing, canvas_drawing])
    w.pack(expand=YES, fill=BOTH)
    w.bind("<B1-Motion>", paint)
    w.bind("<Button-3>", clear_it)

    message = Label(master, text="", font=("", 22), foreground="green")
    message.pack(side=BOTTOM)

    mainloop()


tk_loop()

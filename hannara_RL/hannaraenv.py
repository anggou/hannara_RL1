import tkinter as tk

from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 100  # 픽셀 수
HEIGHT = 5  # 그리드월드 세로
WIDTH = 15  # 그리드월드 가로
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3]  # 좌, 우, 상, 하
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 좌표로 나타낸 행동
REWARDS = []

class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration')
        # 모르겟다 이건
        self.geometry('{0}x{1}'.format(1500, 500 + 50))
        self.env = Env()
        self.agent = agent
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)


        # 그리드 생성


        # 캔버스에 이미지 추가
        # self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        # canvas.create_image(250, 150, image=self.shapes[1])
        # canvas.create_image(150, 250, image=self.shapes[1])
        # canvas.create_image(250, 250, image=self.shapes[2])
        canvas.create_image(250, 125, image=self.shapes[4])
        canvas.create_image(750, 250, image=self.shapes[3])
        for col in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, 1500, row
            canvas.create_line(x0, y0, x1, y1)

        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65, 65)))
        skele = PhotoImage(Image.open("../img/hannara_skele.png").resize((1500, 500)))
        real = PhotoImage(Image.open("../img/hannara_real.png").resize((500, 250)))
        return rectangle, triangle, circle, skele, real


class Env:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.all_state = []

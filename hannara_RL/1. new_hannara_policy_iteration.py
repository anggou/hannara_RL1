import time
import tkinter as tk
from tkinter import Button

import numpy as np
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 60  # 픽셀 수
HEIGHT = 14  # 그리드월드 세로
WIDTH = 25 # 그리드월드 가로
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3]  # 상, 하, 좌, 우
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 좌표로 나타낸 행동 상하좌우
REWARDS = []
maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
rec_lo = [21, 12]
# fire_lo = [10,10]
fire_lo = [5, 12]
liftboat_lo = [3, 7]


# re_fire_lo = [fire_lo[0]+1,fire_lo[1]+1]
# end_state = [6, 5]

class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT + 50))
        self.texts = []
        self.arrows = []
        self.env = Env()
        self.agent = agent
        self.evaluation_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.text_reward(liftboat_lo[0], liftboat_lo[1], "R : 1.0")
        self.text_reward(fire_lo[0], fire_lo[1], "R : -1.0")
        self.text_reward(2, 1, "R : -1.0")

    # 그리드 흰색배경 너비
    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # 버튼 초기화
        # Evaluate 버튼 이름, 대입되는 메소드 변경가능
        iteration_button = Button(self, text="Evaluate",
                                  command=self.evaluate_policy)
        # 버튼 색상변경, 너비 변경
        iteration_button.configure(width=5, activebackground="#33B5E5")
        # iteration_button을 가져와서, 버튼위치 정하고 그리기
        canvas.create_window(WIDTH * UNIT * 0.13, HEIGHT * UNIT + 15,
                             window=iteration_button)

        policy_button = Button(self, text="Improve",
                               command=self.improve_policy)
        policy_button.configure(width=5, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.37, HEIGHT * UNIT + 15,
                             window=policy_button)

        policy_button = Button(self, text="move", command=self.move_by_policy)
        policy_button.configure(width=5, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.62, HEIGHT * UNIT + 15,
                             window=policy_button)

        policy_button = Button(self, text="reset", command=self.reset)
        policy_button.configure(width=5, activebackground="#33B5E5")
        canvas.create_window(WIDTH * UNIT * 0.87, HEIGHT * UNIT + 15,
                             window=policy_button)

        # 그리드 생성
        for col in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, WIDTH * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가 , 여기서는 사각형만 움직이는 물체이기 때문에, self. 형태로 변수선언
        canvas.create_image(WIDTH*UNIT/2, HEIGHT*UNIT/2, image=self.shapes[4])
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if maze[y][x] == 1:
                    canvas.create_image(UNIT * x + (UNIT * 0.5), UNIT * y + (UNIT * 0.5), image=self.shapes[1])
        canvas.create_image(UNIT * fire_lo[0] + (UNIT * 0.5), UNIT * fire_lo[1] + (UNIT * 0.5), image=self.shapes[2])
        self.rectangle = canvas.create_image(UNIT * rec_lo[0] + (UNIT * 0.5), UNIT * rec_lo[1] + (UNIT * 0.5),
                                             image=self.shapes[0])
        canvas.create_image(UNIT * liftboat_lo[0] + (UNIT * 0.5), UNIT * liftboat_lo[1] + (UNIT * 0.5),
                            image=self.shapes[3])

        canvas.pack()

        return canvas

    def load_images(self):
        up = PhotoImage(Image.open("../img/up.png").resize((13, 13)))
        right = PhotoImage(Image.open("../img/right.png").resize((13, 13)))
        left = PhotoImage(Image.open("../img/left.png").resize((13, 13)))
        down = PhotoImage(Image.open("../img/down.png").resize((13, 13)))
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((UNIT - 10, UNIT - 10)))
        block = PhotoImage(Image.open("../img/block.png").resize((UNIT, UNIT)))
        FIRE = PhotoImage(Image.open("../img/FIRE.png").resize((UNIT - 5, UNIT - 5)))
        ship = PhotoImage(Image.open("../img/hannara_skele.png").resize((UNIT*WIDTH, UNIT*HEIGHT)))
        lifeboat = PhotoImage(Image.open("../img/lifeboat.png").resize((UNIT - 5, UNIT - 5)))
        return (up, down, left, right), (rectangle, block, FIRE, lifeboat, ship)

    def reset(self):
        # if self.is_moving == 0:
        self.evaluation_count = 0
        self.improvement_count = 0
        for i in self.texts:
            self.canvas.delete(i)

        for i in self.arrows:
            self.canvas.delete(i)
        self.agent.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]
        self.agent.policy_table = ([[[0.25, 0.25, 0.25, 0.25]] * WIDTH
                                    for _ in range(HEIGHT)])
        self.agent.policy_table[liftboat_lo[1]][liftboat_lo[0]] = []
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, rec_lo[0] * UNIT + UNIT / 2 - x, rec_lo[1] * UNIT + UNIT / 2 - y)

    def text_value(self, row, col, contents, font='Helvetica', size=7,
                   style='normal', anchor="nw"):
        origin_x, origin_y = UNIT - 25, UNIT - 10
        x, y = origin_x + (UNIT * row), origin_y + (UNIT * col)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="red", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def text_reward(self, row, col, contents, font='Helvetica', size=7,
                    style='normal', anchor="nw"):
        origin_x, origin_y = 5, 5
        x, y = origin_x + (UNIT * row), origin_y + (UNIT * col)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def rectangle_move(self, action):
        base_action = np.array([0, 0])
        location = self.find_rectangle()
        self.render()  # 사각형을 모든 작업중에 가장높은 우선순위로 올리기
        if action == 0 and location[1] > 0 and maze[location[1] - 1][location[0]] == 0:  # 상 location[0], y 좌표가 0보다 크고
            base_action[1] -= UNIT
        elif action == 1 and location[1] < HEIGHT - 1 and maze[location[1] + 1][location[0]] == 0:  # 하
            base_action[1] += UNIT
        elif action == 2 and location[0] > 0 and maze[location[1]][location[0] - 1] == 0:  # 좌
            base_action[0] -= UNIT
        elif action == 3 and location[0] < WIDTH - 1 and maze[location[1]][location[0] + 1] == 0:  # 우
            base_action[0] += UNIT
        # move agent
        self.canvas.move(self.rectangle, base_action[0], base_action[1])

    def find_rectangle(self):
        temp = self.canvas.coords(self.rectangle)
        # x = (temp[0] / 100) - 0.5
        # y = (temp[1] / 100) - 0.5
        x = ((temp[0] - UNIT / 2) / UNIT)
        y = ((temp[1] - UNIT / 2) / UNIT)

        return int(x), int(y)

    def move_by_policy(self):
        if self.improvement_count != 0 and self.is_moving != 1:
            self.is_moving = 1
            # 좌표변경
            x, y = self.canvas.coords(self.rectangle)  # 720+30, 660+30
            # # -720 , -660
            self.canvas.move(self.rectangle, UNIT * rec_lo[0] + (UNIT * 0.5) - x, UNIT * rec_lo[1] + (UNIT * 0.5) - y)
            # # 움직이고 바뀐 x,y
            # 정책이 존재하는한, 계쏙 움직이게 해줌.
            x, y = self.find_rectangle()
            while len(self.agent.policy_table[y][x]) != 0:
                self.after(100, self.rectangle_move(self.agent.get_action([x, y])))
                x, y = self.find_rectangle()
                self.canvas.create_text(WIDTH * UNIT * 0.5, HEIGHT * UNIT + 15, fill="red", text=(x, y),
                                        font='Helvetica', anchor="nw")
            self.is_moving = 0

    def draw_one_arrow(self, col, row, policy):
        if col == liftboat_lo[0] and row == liftboat_lo[1]:
            return

        if policy[0] > 0:  # up
            origin_x, origin_y = UNIT / 2 + (UNIT * col), UNIT / 10 + (UNIT * row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.up))
        if policy[1] > 0:  # down
            origin_x, origin_y = UNIT / 2 + (UNIT * col), UNIT * 9 / 10 + (UNIT * row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.down))
        if policy[2] > 0:  # left
            origin_x, origin_y = UNIT / 10 + (UNIT * col), UNIT / 2 + (UNIT * row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.left))
        if policy[3] > 0:  # right
            origin_x, origin_y = UNIT * 9 / 10 + (UNIT * col), UNIT / 2 + (UNIT * row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y,
                                                        image=self.right))

    def draw_from_policy(self, policy_table):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                self.draw_one_arrow(x, y, policy_table[y][x])

    def print_value_table(self, value_table):
        for y in range(HEIGHT):
            for x in range(WIDTH):
                self.text_value(x, y, round(value_table[y][x], 2))

    def render(self):
        time.sleep(0.1)
        # 모든 작업중에 우위로 만들기
        self.canvas.tag_raise(self.rectangle)
        self.update()

    def evaluate_policy(self):
        self.evaluation_count += 1
        for i in self.texts:
            self.canvas.delete(i)
        self.agent.policy_evaluation()
        self.print_value_table(self.agent.value_table)

    def improve_policy(self):
        self.improvement_count += 1
        for i in self.arrows:
            self.canvas.delete(i)
        self.agent.policy_improvement()
        self.draw_from_policy(self.agent.policy_table)


class Env:
    def __init__(self):
        self.transition_probability = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[liftboat_lo[1]][liftboat_lo[0]] = 1  # (2,2) 좌표 동그라미 위치에 보상 1
        self.reward[fire_lo[1]][fire_lo[0]] = -1  # (1,2) 좌표 세모 위치에 보상 -1
        # self.reward[2][1] = -1  # (2,1) 좌표 세모 위치에 보상 -1
        self.all_state = []

        for y in range(HEIGHT):
            for x in range(WIDTH):
                state = [x, y]
                self.all_state.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[1]][next_state[0]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1  # x좌표 0 좌측으로 벗어나면, 0 WIDTH-1 우측으로 벗어나면 4 나머지는 그대로
        if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1
        if state[1] > HEIGHT - 1 else state[1])
        return state

    def get_transition_prob(self, state, action):
        return self.transition_probability

    def get_all_states(self):
        return self.all_state


class PolicyIteration:
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.env = env
        # 가치함수를 2차원 리스트로 초기화
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # 상 하 좌 우 동일한 확률로 정책 초기화

        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                             for _ in range(env.height)]

        # 마침 상태의 설정
        self.policy_table[liftboat_lo[1]][liftboat_lo[0]] = []

        # 할인율
        self.discount_factor = 0.9

    # 벨만 기대 방정식을 통해 다음 가치함수를 계산하는 정책 평가
    def policy_evaluation(self):
        # 다음 가치함수 초기화
        next_value_table = [[0.00] * self.env.width
                            for _ in range(self.env.height)]

        # 모든 상태에 대해서 벨만 기대방정식을 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 마침 상태의 가치 함수 = 0
            if state == [liftboat_lo[0], liftboat_lo[1]]:
                next_value_table[state[1]][state[0]] = value
                continue

            # 벨만 기대 방정식
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] *
                          (reward + self.discount_factor * next_value))

            next_value_table[state[1]][state[0]] = value
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if maze[y][x] == 1 :
                    next_value_table[y][x] = 0.0
        self.value_table = next_value_table

    # 현재 가치 함수에 대해서 탐욕 정책 발전
    ##### state가 어떻게 나오는지 확인
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [liftboat_lo[0], liftboat_lo[1]]:
                continue

            value_list = []
            # 반환할 정책 초기화
            result = [0.0, 0.0, 0.0, 0.0]

            # 모든 행동에 대해서 [보상 + (할인율 * 다음 상태 가치함수)] 계산
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action) #좌우상하
                reward = self.env.get_reward(state, action)  # 다음 상태의 보상을 불러옴
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)

            # 받을 보상이 최대인 행동들에 대해 탐욕 정책 발전
            max_idx_list = np.argwhere(value_list == np.amax(value_list))  # max 값이 어딧는지 , argwhere= ()조건에 맞는 인덱스를 반환
            max_idx_list = max_idx_list.flatten().tolist()  # list 화 시킴.
            prob = 1 / len(max_idx_list)

            for idx in max_idx_list:
                result[idx] = prob
            next_policy[state[1]][state[0]] = result
            # for y in range(HEIGHT):
            #     for x in range(WIDTH):
            #         if maze[y][x] == 1 and maze[y - 1][x] != None:
            #             next_policy[y - 1][x][0] = 0.0
            #         elif maze[y][x] == 1 and maze[y + 1][x] != None:
            #             next_policy[y + 1][x][1] = 0.0
            #         elif maze[y][x] == 1 and maze[y][x - 1] != None:
            #             next_policy[y][x - 1][2] = 0.0
            #         elif maze[y][x] == 1 and maze[y][x + 1] != None:
            #             next_policy[y][x + 1][3] = 0.0

        self.policy_table = next_policy

    # 특정 상태에서 정책에 따라 무작위로 행동을 반환
    def get_action(self, state):
        policy = self.get_policy(state)
        policy = np.array(policy)
        # 4개의 인자중에 한개를 p의 확률로 샘플링한다.  0~3사이의 수를 반환해야하는데 잘모르겠다.
        return np.random.choice(4, 1, p=policy)[0]

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        return self.policy_table[state[1]][state[0]]

    # 가치 함수의 값을 반환
    def get_value(self, state):
        return self.value_table[state[1]][state[0]]


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)  # policy_iteration = agent
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()

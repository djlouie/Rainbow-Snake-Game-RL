# importing libraries
import pygame
import time
import random
import pygame_capture


class Game:
    def __init__(self, x = 720, y = 480, episode = 0):

        # set snake speed
        self.snake_speed = 15

        # set episode
        self.episode = episode
 
        # Window size
        self.window_x = x
        self.window_y = y
        
        # defining colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.orange = pygame.Color(255, 127, 0)
        self.yellow =pygame.Color(255, 255, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)
        self.indigo = pygame.Color(75, 0, 130)
        self.violet = pygame.Color(148, 0, 211)
        self.rainbow = [self.red, self.orange, self.yellow, self.green, self.blue, self.indigo, self.violet]

        # Initialising pygame
        pygame.init()
        
        # Initialise game window
        pygame.display.set_caption(f'GeeksforGeeks Snakes Episode {self.episode}')
        self.game_window = pygame.display.set_mode((self.window_x, self.window_y))
 
        # FPS (frames per second) controller
        self.fps = pygame.time.Clock()
        
        # defining snake default position
        self.snake_position = [100, 240]
        
        # defining first 4 blocks of snake body
        self.snake_body = [[100, 240],
                    [90, 240],
                    [80, 240],
                    [70, 240]
                    ]

        # defining snake colors
        start_color = random.randint(0,6)
        self.snake_colors = [self.rainbow[start_color], self.rainbow[(start_color + 1) % 7], self.rainbow[(start_color + 2) % 7], self.rainbow[(start_color + 3) % 7]]

        # fruit position
        self.fruit_position = [(random.randrange(1, (self.window_x//10)) * 10, random.randrange(1, (self.window_y//10)) * 10),  # red
                        (random.randrange(1, (self.window_x//10)) * 10, random.randrange(1, (self.window_y//10)) * 10),  # orange
                        (random.randrange(1, (self.window_x//10)) * 10, random.randrange(1, (self.window_y//10)) * 10),  # yellow
                        (random.randrange(1, (self.window_x//10)) * 10, random.randrange(1, (self.window_y//10)) * 10),  # green
                        (random.randrange(1, (self.window_x//10)) * 10, random.randrange(1, (self.window_y//10)) * 10),  # blue
                        (random.randrange(1, (self.window_x//10)) * 10, random.randrange(1, (self.window_y//10)) * 10),  # indigo
                        (random.randrange(1, (self.window_x//10)) * 10, random.randrange(1, (self.window_y//10)) * 10),  # violet
                        ]
        
        self.red_fruit_spawn = True
        self.orange_fruit_spawn = True
        self.yellow_fruit_spawn = True
        self.green_fruit_spawn = True
        self.blue_fruit_spawn = True
        self.indigo_fruit_spawn = True
        self.violet_fruit_spawn = True
 
        # setting default snake direction towards
        # right
        self.direction = 'RIGHT'
        self.change_to = self.direction
        
        # initial score
        self.score = 0

        # terminated variable
        self.terminated = False
 
    # displaying Score function
    def show_score(self, choice, color, font, size):
    
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)
        
        # create the display surface object 
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        
        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        
        # displaying text
        self.game_window.blit(score_surface, score_rect)

    # game over function
    def game_over(self):
    
        # creating font object my_font
        my_font = pygame.font.SysFont('times new roman', 50)
        
        # creating a text surface on which text 
        # will be drawn
        game_over_surface = my_font.render(
            'Your Score is : ' + str(self.score), True, self.red)
        
        # create a rectangular object for the text 
        # surface object
        game_over_rect = game_over_surface.get_rect()
        
        # setting position of the text
        game_over_rect.midtop = (self.window_x/2, self.window_y/4)
        
        # blit will draw the text on screen
        self.game_window.blit(game_over_surface, game_over_rect)
        pygame.display.flip()
        
        # after 2 seconds we will quit the program
        # time.sleep(0.5)
        
        # deactivating pygame library
        pygame.quit()
        
        # quit the program
        # quit()

    # def handle_events(self):
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             self.running = False

    def render(self):
        self.game_window.fill(self.black)
        
        for pos, color in zip(self.snake_body, self.snake_colors):
            pygame.draw.rect(self.game_window, color,
                            pygame.Rect(pos[0], pos[1], 10, 10))
        
        for num, fruit in enumerate(self.fruit_position):
            pygame.draw.rect(self.game_window, self.rainbow[num % 7], pygame.Rect(
                fruit[0], fruit[1], 10, 10))
    
        # displaying score continuously
        self.show_score(1, self.white, 'times new roman', 20)
    
        # Refresh game screen
        pygame.display.update()
    
        # Frame Per Second /Refresh Rate
        self.fps.tick(self.snake_speed)

    # Main Function
    # action is a 0, 1, 2, or 3 which correspond to up, down, right left
    def step(self, action=None):

        if action is None:
            # handling key events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        self.change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        self.change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        self.change_to = 'RIGHT'
        else:
            if action == 0:
                self.change_to = 'UP'
            if action == 1:
                self.change_to = 'DOWN'
            if action == 2:
                self.change_to = 'LEFT'
            if action == 3:
                self.change_to = 'RIGHT'

        # If two keys pressed simultaneously
        # we don't want snake to move into two 
        # directions simultaneously
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

    
        # Moving the snake
        if self.direction == 'UP':
            self.snake_position[1] -= 10
        if self.direction == 'DOWN':
            self.snake_position[1] += 10
        if self.direction == 'LEFT':
            self.snake_position[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_position[0] += 10
        
        # Snake body growing mechanism
        # if fruits and snakes collide then scores
        # will be incremented by 10
        self.snake_body.insert(0, list(self.snake_position))
        if self.snake_position[0] == self.fruit_position[0][0] and self.snake_position[1] == self.fruit_position[0][1]:
            self.score += 10
            self.snake_colors.append(self.red)
            self.red_fruit_spawn = False
        elif self.snake_position[0] == self.fruit_position[1][0] and self.snake_position[1] == self.fruit_position[1][1]:
            self.score += 10
            self.snake_colors.append(self.orange)
            self.orange_fruit_spawn = False
        elif self.snake_position[0] == self.fruit_position[2][0] and self.snake_position[1] == self.fruit_position[2][1]:
            self.score += 10
            self.snake_colors.append(self.yellow)
            self.yellow_fruit_spawn = False
        elif self.snake_position[0] == self.fruit_position[3][0] and self.snake_position[1] == self.fruit_position[3][1]:
            self.score += 10
            self.snake_colors.append(self.green)
            self.green_fruit_spawn = False
        elif self.snake_position[0] == self.fruit_position[4][0] and self.snake_position[1] == self.fruit_position[4][1]:
            self.score += 10
            self.snake_colors.append(self.blue)
            self.blue_fruit_spawn = False
        elif self.snake_position[0] == self.fruit_position[5][0] and self.snake_position[1] == self.fruit_position[5][1]:
            self.score += 10
            self.snake_colors.append(self.indigo)
            self.indigo_fruit_spawn = False
        elif self.snake_position[0] == self.fruit_position[6][0] and self.snake_position[1] == self.fruit_position[6][1]:
            self.score += 10
            self.snake_colors.append(self.violet)
            self.violet_fruit_spawn = False
        else:
            self.snake_body.pop()
            
        if not self.red_fruit_spawn:
            self.fruit_position[0] = (random.randrange(1, (self.window_x//10)) * 10, 
                                random.randrange(1, (self.window_y//10)) * 10)
        elif not self.orange_fruit_spawn:
            self.fruit_position[1] = (random.randrange(1, (self.window_x//10)) * 10, 
                                random.randrange(1, (self.window_y//10)) * 10)
        elif not self.yellow_fruit_spawn:
            self.fruit_position[2] = (random.randrange(1, (self.window_x//10)) * 10, 
                                random.randrange(1, (self.window_y//10)) * 10)
        elif not self.green_fruit_spawn:
            self.fruit_position[3] = (random.randrange(1, (self.window_x//10)) * 10, 
                                random.randrange(1, (self.window_y//10)) * 10)
        elif not self.blue_fruit_spawn:
            self.fruit_position[4] = (random.randrange(1, (self.window_x//10)) * 10, 
                                random.randrange(1, (self.window_y//10)) * 10)
        elif not self.indigo_fruit_spawn:
            self.fruit_position[5] = (random.randrange(1, (self.window_x//10)) * 10, 
                                random.randrange(1, (self.window_y//10)) * 10)
        elif not self.violet_fruit_spawn:
            self.fruit_position[6] = (random.randrange(1, (self.window_x//10)) * 10, 
                                random.randrange(1, (self.window_y//10)) * 10)
            
        self.red_fruit_spawn = True
        self.orange_fruit_spawn = True
        self.yellow_fruit_spawn = True
        self.green_fruit_spawn = True
        self.blue_fruit_spawn = True
        self.indigo_fruit_spawn = True
        self.violet_fruit_spawn = True

        # Game Over conditions
        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_position[0] == block[0] and self.snake_position[1] == block[1]:
                self.game_over()
                self.terminated = True
                return self.terminated
        
        # Out of bounds
        if self.snake_position[0] < 0 or self.snake_position[0] > self.window_x-10:
            self.game_over()
            self.terminated = True
            return self.terminated
        if self.snake_position[1] < 0 or self.snake_position[1] > self.window_y-10:
            self.game_over()
            self.terminated = True
            return self.terminated
        
        self.render()

        return self.terminated

        

if __name__ == "__main__":

    game = Game()

    # Initialize the recorder
    recorder = pygame_capture.Recorder(game.game_window, 'output.mp4', 'MJPG')

    # Start recording
    recorder.start()

    while True:
        terminated = game.step()
        if terminated:
            break
    
    # Stop recording
    recorder.stop()

    # Close the recorder
    recorder.close()

    exit
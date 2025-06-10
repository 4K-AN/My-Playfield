import pygame
import random
import sys

# Inisialisasi PyGame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("UFO Penghancur Asteroid")
clock = pygame.time.Clock()

# Warna
BACKGROUND = (10, 5, 30)
PLAYER_COLOR = (0, 255, 200)
ASTEROID_COLOR = (255, 100, 50)
LASER_COLOR = (0, 200, 255)
TEXT_COLOR = (100, 255, 150)

# Player (UFO)
player_size = 50
player_x = WIDTH // 2 - player_size // 2
player_y = HEIGHT - 100
player_speed = 7

# Laser
lasers = []
laser_speed = 10

# Asteroid
asteroids = []
asteroid_speed = 3
asteroid_spawn_rate = 30  # Semakin kecil, semakin sering

# Skor
score = 0
font = pygame.font.SysFont('Arial', 28)

def draw_player(x, y):
    pygame.draw.circle(screen, PLAYER_COLOR, (x, y), player_size//2)
    pygame.draw.circle(screen, (255, 255, 255), (x, y), player_size//3, 1)

def draw_laser(x, y):
    pygame.draw.rect(screen, LASER_COLOR, (x, y, 4, 15))

def draw_asteroid(x, y, size):
    pygame.draw.circle(screen, ASTEROID_COLOR, (x, y), size)

def show_score():
    score_text = font.render(f'Skor: {score}', True, TEXT_COLOR)
    screen.blit(score_text, (10, 10))

def game_over():
    game_over_text = font.render('GAME OVER! Tekan R untuk restart', True, TEXT_COLOR)
    screen.blit(game_over_text, (WIDTH//2 - 200, HEIGHT//2))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
    return False

# Main game loop
running = True
game_active = True

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and game_active:
            if event.key == pygame.K_SPACE:
                lasers.append([player_x - 2, player_y - 25])
    
    if game_active:
        # Player movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and player_x > player_size//2:
            player_x -= player_speed
        if keys[pygame.K_RIGHT] and player_x < WIDTH - player_size//2:
            player_x += player_speed

        # Laser movement
        for laser in lasers[:]:
            laser[1] -= laser_speed
            if laser[1] < 0:
                lasers.remove(laser)

        # Spawn asteroid
        if random.randint(1, asteroid_spawn_rate) == 1:
            asteroid_size = random.randint(20, 50)
            asteroid_x = random.randint(asteroid_size, WIDTH - asteroid_size)
            asteroids.append([asteroid_x, -asteroid_size, asteroid_size])

        # Asteroid movement
        for asteroid in asteroids[:]:
            asteroid[1] += asteroid_speed
            if asteroid[1] > HEIGHT + asteroid[2]:
                asteroids.remove(asteroid)
                
                # Game over condition
                if asteroid[1] > HEIGHT:
                    game_active = False

        # Collision detection
        for laser in lasers[:]:
            for asteroid in asteroids[:]:
                distance = ((laser[0] - asteroid[0])**2 + (laser[1] - asteroid[1])**2)**0.5
                if distance < asteroid[2]:
                    if laser in lasers:
                        lasers.remove(laser)
                    if asteroid in asteroids:
                        asteroids.remove(asteroid)
                    score += 10
                    break

        # Player-asteroid collision
        for asteroid in asteroids[:]:
            distance = ((player_x - asteroid[0])**2 + (player_y - asteroid[1])**2)**0.5
            if distance < (player_size//2 + asteroid[2]):
                game_active = False

        # Rendering
        screen.fill(BACKGROUND)
        
        # Bintang latar belakang
        for _ in range(50):
            pygame.draw.circle(screen, (200, 200, 255), 
                              (random.randint(0, WIDTH), random.randint(0, HEIGHT)), 
                              1)
        
        draw_player(player_x, player_y)
        
        for laser in lasers:
            draw_laser(laser[0], laser[1])
            
        for asteroid in asteroids:
            draw_asteroid(asteroid[0], asteroid[1], asteroid[2])
        
        show_score()
    
    else:
        # Restart game
        if game_over():
            player_x = WIDTH // 2 - player_size // 2
            player_y = HEIGHT - 100
            lasers = []
            asteroids = []
            score = 0
            game_active = True

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
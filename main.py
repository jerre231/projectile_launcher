import pygame
import sys

pygame.init()

def main():

    # Display setup
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("temp") # TODO: define game name

    # Set colors theme
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # Runtime variables
    running = True
    clock = pygame.time.Clock()

    # Game itself
    while running:
        
        # Quit handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        # Screen elements
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLACK, (width//2 -50, height//2-50, 100, 100))

        # Display update
        pygame.display.flip()

        # Frame rate
        clock.tick(60)

    # Quit
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
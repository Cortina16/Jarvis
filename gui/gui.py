import pygame
import sys

pygame.font.init()

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 163, 255)
NAVY_BLUE = (0, 0, 128)
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("J.A.R.V.I.S.")
clock = pygame.time.Clock()


class Frame:

    #TODO: add more colors to the glow color list

        
    def __init__(self, x, y, width, height, color, glow_strength = 10, glow_color=(0,0,0,0)):
        """
        crates a new frame object
        :param x: x position, where origin is at the top-left corner
        :param y: y position, where origin is at the top-left corner
        :param width: dimension of the frame
        :param height: height dimension of the frame
        :param color: the main background color
        :param glow_strength: the glow strength
        :param glow_color: the color of the border glow
        """
        self.GLOW_COLORS = []
        for i in range(40):
            self.GLOW_COLORS.append(tuple(list(glow_color)[:3]+[glow_color[3]*i]))
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.glow_strength = glow_strength
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.shrink_factor = 0.5 #how close together the gradient should be
        
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(self.surface, self.color, self.surface.get_rect(), border_radius=10)
        
        self.glow_surface = pygame.Surface((width + 2 * glow_strength, height + 2 * glow_strength), pygame.SRCALPHA)
        for i, (r,g,b, alpha) in enumerate(self.GLOW_COLORS):
            glow_rect = pygame.Rect(
                (i*self.shrink_factor), (i*self.shrink_factor),
                self.glow_surface.get_width() -(2 * i * self.shrink_factor),
                self.glow_surface.get_height() -(2 * i * self.shrink_factor)
            )
            pygame.draw.rect(self.glow_surface, (r, g, b, alpha), glow_rect, border_radius=10)
            
    def draw(self, screen):
        screen.blit(self.glow_surface, (self.rect.x-self.glow_strength, self.rect.y-self.glow_strength))
        screen.blit(self.surface, self.rect)
        
class Label:
    def __init__(self, x, y, text, color, size = 30):
        """
        Creates a text label at given cords
        :param x: x position, where origin is at the top-left corner
        :param y: y position, where origin is at the top-left corner
        :param text: The text to be displayed
        :param color: color of the text.
        """
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.size = size
        font = pygame.font.SysFont("Arial", size=self.size)
        self.text_surface = font.render(self.text, True, self.color)
    def draw(self, screen):
        
        screen.blit(self.text_surface, (self.x, self.y))
        
        
    
    
        
        

def main_loop():
    running = True
    frame = Frame(1000, 500, 200, 300, (0, 0, 0), 10, glow_color=(217, 1, 102, 5))
    frame.draw(screen)
    welcome_message = Label(x=100, y=100, text="J.A.R.V.I.S. ONLINE", color=WHITE, size=72)
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- TODO: Add custom input handling here ---
        # --- TODO: Add UI drawing logic here ---
        frame.draw(screen)
        welcome_message.draw(screen)


        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_loop()
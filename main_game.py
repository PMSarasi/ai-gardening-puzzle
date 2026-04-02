import pygame
import sys
import torch
import numpy as np
import os
from environment.garden_env_normalized import GardenEnvNormalized
from agents.dqn_agent_fixed import DQNAgentFixed

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 750
GRID_SIZE = 3
CELL_SIZE = 140
INFO_HEIGHT = 180

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
LIGHT_GREEN = (200, 230, 200)
DARK_GREEN = (0, 100, 0)
BROWN = (139, 69, 19)
BLUE = (100, 149, 237)
LIGHT_BLUE = (173, 216, 230)
RED = (220, 20, 60)
YELLOW = (255, 215, 0)
GOLD = (255, 215, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
PURPLE = (147, 112, 219)
PINK = (255, 192, 203)

class AIGardeningGameEnhanced:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🌱 AI Gardening Puzzle - Intelligent Plant Care")
        
        # Load fonts
        self.title_font = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.big_font = pygame.font.Font(None, 36)
        
        # Game state
        self.env = GardenEnvNormalized(grid_size=GRID_SIZE)
        self.agent = None
        self.load_agent()
        
        self.mode = "AI"  # AI or HUMAN
        self.running = True
        self.auto_play = False
        self.auto_step = 0
        self.current_action = None
        self.selected_plant = 0
        self.actions = ["💧 Water", "🌿 Fertilize", "✂️ Prune", "⏰ Wait"]
        self.selected_action = 0
        
        self.reset_game()
        
        # Animation
        self.animation_timer = 0
        self.show_action_feedback = None
        self.feedback_timer = 0
        
        # Background
        self.background = self.create_background()
        
    def create_background(self):
        """Create gradient background"""
        bg = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        for y in range(SCREEN_HEIGHT):
            color_value = 135 + int(120 * (y / SCREEN_HEIGHT))
            color = (135, 206, min(235, color_value))
            pygame.draw.line(bg, color, (0, y), (SCREEN_WIDTH, y))
        return bg
    
    def load_agent(self):
        """Load trained model"""
        try:
            state_size = 57
            action_size = 36
            device = torch.device("cpu")
            self.agent = DQNAgentFixed(state_size, action_size, device)
            if os.path.exists("models/final_normalized.pth"):
                self.agent.load("models/final_normalized.pth")
                self.agent.epsilon = 0.01
                print("✅ Agent loaded successfully!")
                return True
            else:
                print("⚠️ Model not found")
                return False
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False
    
    def reset_game(self):
        """Reset the garden"""
        self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.auto_play = False
        self.show_action_feedback = None
        
    def get_plant_sprite(self, plant):
        """Get plant sprite based on type and growth stage"""
        growth = plant.growth_stage
        plant_type = plant.type.name
        
        # Growth stage determines sprite
        if growth < 0.8:
            return "🌱"  # Seedling
        elif growth < 1.6:
            if plant_type == 'FLOWER':
                return "🌿"  # Sprout
            elif plant_type == 'VEGETABLE':
                return "🥬"  # Leafy green
            else:
                return "🌲"  # Small tree
        elif growth < 2.4:
            if plant_type == 'FLOWER':
                return "🌸"  # Flower bud
            elif plant_type == 'VEGETABLE':
                return "🍅"  # Growing vegetable
            else:
                return "🌳"  # Growing tree
        else:
            if plant_type == 'FLOWER':
                return "🌻"  # Full flower
            elif plant_type == 'VEGETABLE':
                return "🥕"  # Ripe vegetable
            else:
                return "🎄"  # Mature tree
    
    def get_health_color(self, health):
        """Get color based on health"""
        if health > 70:
            return (50, 205, 50)  # Lime green
        elif health > 40:
            return (255, 165, 0)  # Orange
        else:
            return (220, 20, 60)  # Red
    
    def draw_plant_glow(self, screen, x, y, health):
        """Draw glow effect based on health"""
        if health > 80:
            glow_color = (50, 205, 50, 100)
            radius = 35
        elif health > 60:
            glow_color = (255, 215, 0, 80)
            radius = 30
        elif health > 40:
            glow_color = (255, 165, 0, 60)
            radius = 25
        else:
            glow_color = (220, 20, 60, 50)
            radius = 20
        
        glow_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (radius, radius), radius)
        screen.blit(glow_surf, (x - radius, y - radius))
    
    def draw_garden(self):
        """Draw beautiful garden grid"""
        margin_x = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
        margin_y = 50
        
        # Draw grass background for each cell
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = margin_x + j * CELL_SIZE
                y = margin_y + i * CELL_SIZE
                
                # Draw cell with gradient
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                for k in range(CELL_SIZE):
                    color_value = 200 - int(50 * (k / CELL_SIZE))
                    pygame.draw.line(self.screen, (color_value, 230, color_value), 
                                   (x, y + k), (x + CELL_SIZE, y + k))
                
                # Draw border
                pygame.draw.rect(self.screen, DARK_GREEN, rect, 3)
                
                # Draw decorative corners
                corner_size = 10
                for corner in [(x, y), (x + CELL_SIZE - corner_size, y), 
                              (x, y + CELL_SIZE - corner_size), 
                              (x + CELL_SIZE - corner_size, y + CELL_SIZE - corner_size)]:
                    pygame.draw.rect(self.screen, GOLD, (*corner, corner_size, corner_size))
        
        # Draw plants
        for idx, plant in enumerate(self.env.plants):
            i = idx // GRID_SIZE
            j = idx % GRID_SIZE
            
            x = margin_x + j * CELL_SIZE + CELL_SIZE // 2
            y = margin_y + i * CELL_SIZE + CELL_SIZE // 2 - 15
            
            # Draw glow effect based on health
            self.draw_plant_glow(self.screen, x, y - 10, plant.health)
            
            # Draw plant emoji
            emoji = self.get_plant_sprite(plant)
            emoji_font = pygame.font.Font(None, 72)
            emoji_surface = emoji_font.render(emoji, True, BLACK)
            emoji_rect = emoji_surface.get_rect(center=(x, y - 10))
            self.screen.blit(emoji_surface, emoji_rect)
            
            # Draw health bar
            health_bar_width = CELL_SIZE - 30
            health_bar_height = 12
            health_bar_x = x - health_bar_width // 2
            health_bar_y = y + 30
            
            health_percent = plant.health / 100
            health_color = self.get_health_color(plant.health)
            
            # Background bar
            pygame.draw.rect(self.screen, (100, 100, 100), 
                           (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
            # Health bar
            pygame.draw.rect(self.screen, health_color, 
                           (health_bar_x, health_bar_y, health_bar_width * health_percent, health_bar_height))
            
            # Draw resource indicators
            # Water drop
            water_color = BLUE if plant.water > 2 else (100, 149, 237) if plant.water > 1 else (70, 130, 180)
            pygame.draw.circle(self.screen, water_color, (x - 35, y + 45), 12)
            water_text = self.small_font.render(f"{plant.water:.0f}", True, WHITE)
            water_rect = water_text.get_rect(center=(x - 35, y + 45))
            self.screen.blit(water_text, water_rect)
            
            # Soil
            soil_color = BROWN if plant.soil > 2 else (160, 82, 45) if plant.soil > 1 else (139, 69, 19)
            pygame.draw.rect(self.screen, soil_color, (x + 20, y + 35, 24, 24))
            soil_text = self.small_font.render(f"{plant.soil:.0f}", True, WHITE)
            soil_rect = soil_text.get_rect(center=(x + 32, y + 47))
            self.screen.blit(soil_text, soil_rect)
            
            # Growth indicator
            growth_text = self.small_font.render(f"🌱{plant.growth_stage:.1f}", True, DARK_GREEN)
            self.screen.blit(growth_text, (x - 15, y + 55))
            
            # Highlight selected plant in human mode
            if self.mode == "HUMAN" and self.selected_plant == idx:
                pygame.draw.rect(self.screen, GOLD, 
                               (margin_x + j * CELL_SIZE, margin_y + i * CELL_SIZE, 
                                CELL_SIZE, CELL_SIZE), 4)
    
    def draw_info_panel(self):
        """Draw beautiful info panel"""
        panel_y = SCREEN_HEIGHT - INFO_HEIGHT
        
        # Semi-transparent panel
        panel_surf = pygame.Surface((SCREEN_WIDTH, INFO_HEIGHT))
        panel_surf.set_alpha(220)
        panel_surf.fill((50, 50, 50))
        self.screen.blit(panel_surf, (0, panel_y))
        
        # Top border
        pygame.draw.line(self.screen, GOLD, (0, panel_y), (SCREEN_WIDTH, panel_y), 3)
        
        # Garden Stats (Left)
        stats = self.env.get_garden_stats()
        
        # Health with icon
        health_color = self.get_health_color(stats['avg_health'])
        health_text = self.font.render(f"❤️ Health: {stats['avg_health']:.1f}%", True, health_color)
        self.screen.blit(health_text, (20, panel_y + 15))
        
        # Other stats
        stats_texts = [
            f"📈 Growth: {stats['avg_growth']:.1f}/3",
            f"💧 Water: {stats['avg_water']:.1f}/5",
            f"🌍 Soil: {stats['avg_soil']:.1f}/5",
            f"📅 Day: {self.step_count}/100"
        ]
        
        y_offset = panel_y + 50
        for text in stats_texts:
            stat_surface = self.small_font.render(text, True, WHITE)
            self.screen.blit(stat_surface, (20, y_offset))
            y_offset += 25
        
        # Score (Center)
        score_text = self.title_font.render(f"🏆 {self.total_reward:.0f}", True, GOLD)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, panel_y + 60))
        self.screen.blit(score_text, score_rect)
        
        # Mode indicator
        mode_text = "🤖 AI MODE" if self.mode == "AI" else "👤 HUMAN MODE"
        mode_color = BLUE if self.mode == "AI" else GREEN
        mode_surface = self.big_font.render(mode_text, True, mode_color)
        mode_rect = mode_surface.get_rect(center=(SCREEN_WIDTH // 2, panel_y + 20))
        self.screen.blit(mode_surface, mode_rect)
        
        # Controls (Right)
        controls_x = SCREEN_WIDTH - 280
        controls_y = panel_y + 15
        
        if self.mode == "AI":
            controls = [
                "🎮 AI CONTROLS:",
                "  SPACE - Single Move",
                "  A - Auto Play ON/OFF",
                "  H - Switch to Human",
                "  R - Reset Garden",
                "  ESC - Exit"
            ]
            if self.auto_play:
                auto_surface = self.font.render("🔁 AUTO PLAY ACTIVE", True, YELLOW)
                self.screen.blit(auto_surface, (controls_x, controls_y + 130))
        else:
            controls = [
                "🎮 HUMAN CONTROLS:",
                f"  ←/→ - Select Plant ({self.selected_plant + 1}/9)",
                f"  ↑/↓ - Select Action ({self.actions[self.selected_action]})",
                "  ENTER - Take Action",
                "  A - Switch to AI Mode",
                "  R - Reset Garden"
            ]
        
        for i, control in enumerate(controls):
            control_surface = self.small_font.render(control, True, WHITE)
            self.screen.blit(control_surface, (controls_x, controls_y + i * 22))
        
        # Auto play indicator
        if self.mode == "AI" and self.auto_play:
            indicator_x = SCREEN_WIDTH - 100
            indicator_y = panel_y + INFO_HEIGHT - 30
            pygame.draw.circle(self.screen, GREEN, (indicator_x, indicator_y), 8)
            text = self.small_font.render("AUTO", True, WHITE)
            self.screen.blit(text, (indicator_x - 25, indicator_y - 8))
    
    def draw_action_feedback(self):
        """Draw action feedback popup"""
        if self.show_action_feedback and self.feedback_timer > 0:
            self.feedback_timer -= 1
            
            # Create popup
            popup_surf = pygame.Surface((300, 60), pygame.SRCALPHA)
            popup_surf.fill((0, 0, 0, 200))
            
            text = self.font.render(self.show_action_feedback, True, GOLD)
            text_rect = text.get_rect(center=(150, 30))
            popup_surf.blit(text, text_rect)
            
            self.screen.blit(popup_surf, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 100))
    
    def ai_move(self):
        """Perform one AI move"""
        if self.agent:
            state = self.env._get_observation()
            action = self.agent.act(state, eval_mode=True)
            next_state, reward, done, truncated, _ = self.env.step(action)
            self.total_reward += reward
            self.step_count += 1
            
            # Show action feedback
            action_names = ["Water", "Fertilize", "Prune", "Wait"]
            plant_idx = action // 4
            action_type = action % 4
            self.show_action_feedback = f"🤖 AI {action_names[action_type]} Plant {plant_idx + 1} (+{reward:.1f})"
            self.feedback_timer = 30
            
            if done or truncated:
                self.show_completion_message()
                return True
        return False
    
    def human_move(self):
        """Perform human move"""
        action = self.selected_plant * 4 + self.selected_action
        next_state, reward, done, truncated, _ = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        
        # Show action feedback
        self.show_action_feedback = f"👤 You {self.actions[self.selected_action]} (+{reward:.1f})"
        self.feedback_timer = 30
        
        return done or truncated
    
    def show_completion_message(self):
        """Show episode completion message"""
        stats = self.env.get_garden_stats()
        final_score = self.total_reward
        
        # Create overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Success message
        if final_score > 70:
            title = "🎉 GARDEN MASTER! 🎉"
            color = GOLD
        elif final_score > 50:
            title = "🌱 GOOD GARDENER! 🌱"
            color = GREEN
        else:
            title = "🌿 KEEP LEARNING! 🌿"
            color = BLUE
        
        title_surface = self.title_font.render(title, True, color)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 80))
        self.screen.blit(title_surface, title_rect)
        
        score_surface = self.big_font.render(f"Final Score: {final_score:.0f}", True, WHITE)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 30))
        self.screen.blit(score_surface, score_rect)
        
        health_surface = self.font.render(f"Garden Health: {stats['avg_health']:.1f}%", True, WHITE)
        health_rect = health_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 10))
        self.screen.blit(health_surface, health_rect)
        
        growth_surface = self.font.render(f"Growth Stage: {stats['avg_growth']:.1f}/3", True, WHITE)
        growth_rect = growth_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 40))
        self.screen.blit(growth_surface, growth_rect)
        
        continue_surface = self.small_font.render("Press SPACE to continue...", True, GRAY)
        continue_rect = continue_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 100))
        self.screen.blit(continue_surface, continue_rect)
        
        pygame.display.flip()
        
        # Wait for space
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False
                    self.reset_game()
                elif event.type == pygame.QUIT:
                    waiting = False
                    self.running = False
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        self.show_action_feedback = "🔄 Garden Reset!"
                        self.feedback_timer = 30
                    
                    elif event.key == pygame.K_a:
                        if self.mode == "AI":
                            self.mode = "HUMAN"
                            self.show_action_feedback = "👤 Switched to HUMAN Mode"
                        else:
                            self.mode = "AI"
                            self.auto_play = False
                            self.show_action_feedback = "🤖 Switched to AI Mode"
                        self.feedback_timer = 30
                    
                    elif self.mode == "AI":
                        if event.key == pygame.K_SPACE:
                            self.ai_move()
                        elif event.key == pygame.K_TAB:
                            self.auto_play = not self.auto_play
                            if self.auto_play:
                                self.show_action_feedback = "🔁 AUTO PLAY ON"
                            else:
                                self.show_action_feedback = "⏸️ AUTO PLAY OFF"
                            self.feedback_timer = 30
                    
                    elif self.mode == "HUMAN":
                        if event.key == pygame.K_LEFT:
                            self.selected_plant = (self.selected_plant - 1) % 9
                        elif event.key == pygame.K_RIGHT:
                            self.selected_plant = (self.selected_plant + 1) % 9
                        elif event.key == pygame.K_UP:
                            self.selected_action = (self.selected_action - 1) % 4
                        elif event.key == pygame.K_DOWN:
                            self.selected_action = (self.selected_action + 1) % 4
                        elif event.key == pygame.K_RETURN:
                            finished = self.human_move()
                            if finished:
                                self.show_completion_message()
            
            # Auto play
            if self.mode == "AI" and self.auto_play:
                finished = self.ai_move()
                if finished:
                    self.auto_play = False
                pygame.time.wait(300)
            
            # Draw everything
            self.screen.blit(self.background, (0, 0))
            self.draw_garden()
            self.draw_info_panel()
            self.draw_action_feedback()
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = AIGardeningGameEnhanced()
    game.run()
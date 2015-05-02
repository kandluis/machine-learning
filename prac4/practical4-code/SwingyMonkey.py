import sys
import pygame as pg
import numpy.random as npr

# universal file for game parameters
from parameters import game_params

class SwingyMonkey:

    def __init__(self,visual=True,sound=True, text=None, action_callback=None, 
                 reward_callback=None, tick_length=100):
        """Constructor for the SwingyMonkey class.

        Possible Keyword Arguments:

        visual: Boolean variable on whether or not to visualize the game.
                Default to True. False leads to quicker iterations.
                False argument also renders tick_length useless, as no time is
                spent waiting between game_steps. The game window is still displayed,
                but no images are rendered.

        sound: Boolean variable on whether or not to play sounds.
               Defaults to True.

        text: Optional string to display in the upper right corner of
              the screen.

        action_callback: Function handle for determining actions.
                         Takes a dictionary as an argument.  The
                         dictionary contains the current state of the
                         game.

        reward_callback: Function handle for receiving rewards. Takes
                         a scalar argument which is the reward.

        tick_length: Time in milliseconds between game steps.
                     Defaults to 100ms, but you might want to make it
                     smaller for training."""

        # Specify screen rendering properties
        self.visual = visual

        # Don't change these.
        self.screen_width  = game_params['screen_width']
        self.screen_height = game_params['screen_height']
        self.horz_speed    = game_params['horz_speed']
        self.impulse       = game_params['impulse']
        self.gravity       = game_params['gravity']
        self.tree_mean     = game_params['tree_mean']
        self.tree_gap      = game_params['tree_gap']
        self.tree_offset   = game_params['tree_offset']
        self.edge_penalty  = game_params['edge_penalty']
        self.tree_penalty  = game_params['tree_penalty']
        self.tree_reward   = game_params['tree_reward']

        # Store arguments.
        self.sound         = sound
        self.action_fn     = action_callback
        self.reward_fn     = reward_callback
        self.tick_length   = tick_length
        self.text          = text

        # Initialize pygame.
        pg.init()
        try:
            pg.mixer.init()
        except:
            print "No sound."
            self.sound = False

        # Set up the screen for rendering.

        self.screen = pg.display.set_mode((self.screen_width, self.screen_height), 0, 32)

        # Load external resources.
        self.background_img = pg.image.load('res/jungle-pixel.png').convert()
        self.monkey_img     = pg.image.load('res/monkey.png').convert_alpha()
        self.tree_img       = pg.image.load('res/tree-pixel.png').convert_alpha()
        if self.sound:
            self.screech_snd    = pg.mixer.Sound('res/screech.wav')
            self.blop_snd       = pg.mixer.Sound('res/blop.wav')

        # Set up text rendering.
        self.font = pg.font.Font(None, 36)

        # Track locations of trees and gaps.
        self.trees     = []
        self.next_tree = 0
        
        # Precompute some things about the monkey.
        self.monkey_left  = self.screen_width/2 - self.monkey_img.get_width()/2
        self.monkey_right = self.monkey_left + self.monkey_img.get_width()
        self.monkey_loc   = self.screen_height/2 - self.monkey_img.get_height()/2

        # Track game state.
        self.vel   = 0
        self.hook  = self.screen_width
        self.score = 0
        self.iter  = 0

    def get_state(self):
        '''Returns a snapshot of the current game state, computed
        relative to to the next oncoming tree.  This is a dictionary
        with the following structure:
        { 'score': <current score>,
          'tree': { 'dist': <pixels to next tree trunk>,
                    'top':  <screen height of top of tree trunk gap>,
                    'bot':  <screen height of bottom of tree trunk gap> },
          'monkey': { 'vel': <current monkey y-axis speed in pixels per iteration>,
                      'top': <screen height of top of monkey>,
                      'bot': <screen height of bottom of monkey> }}'''                      

        # Find the next closest tree.
        for tree in self.trees:
            if tree['x']+290 > self.monkey_left:
                next_tree = tree.copy()
                break

        # Construct the state dictionary to return.
        return { 'score': self.score,
                 'tree': { 'dist': next_tree['x']+215-self.monkey_right,
                           'top': self.screen_height-next_tree['y'],
                           'bot': self.screen_height-next_tree['y']-self.tree_gap},
                 'monkey': { 'vel': self.vel,
                             'top': self.screen_height - self.monkey_loc + self.monkey_img.get_height()/2,
                             'bot': self.screen_height - self.monkey_loc - self.monkey_img.get_height()/2}}

    def game_loop(self):
        '''This is called every game tick.  You call this in a loop
        until it returns false, which means you hit a tree trunk, fell
        off the bottom of the screen, or jumped off the top of the
        screen.  It calls the action and reward callbacks.'''

        # Render the background.
        if self.visual:
            self.screen.blit(self.background_img, (self.iter,0))
            if self.iter < self.background_img.get_width() - self.screen_width:
                self.screen.blit(self.background_img, (self.iter+self.background_img.get_width(),0))

        # Perhaps generate a new tree.
        if self.next_tree <= 0:
            self.next_tree = self.tree_img.get_width() + int(npr.geometric(1.0/self.tree_mean))
            self.trees.append( { 'x': self.screen_width+1,
                                 'y': int((0.3 + npr.rand()*0.65)*(self.screen_height-self.tree_gap)),
                                 's': False })
        # Process input events.
        if self.visual:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    sys.exit()
                elif self.action_fn is None and event.type == pg.KEYDOWN:
                    self.vel = npr.poisson(self.impulse)
                    self.hook = self.screen_width

        # Perhaps take an action via the callback.
        if self.action_fn is not None and self.action_fn(self.get_state()):
            self.vel = npr.poisson(self.impulse)
            self.hook = self.screen_width

        # Eliminate trees that have moved off the screen.
        self.trees = filter(lambda x: x['x'] > -self.tree_img.get_width(), self.trees)

        # Monkey dynamics
        self.monkey_loc -= self.vel
        self.vel        -= self.gravity

        # Current monkey bounds.
        monkey_top = self.monkey_loc - self.monkey_img.get_height()/2
        monkey_bot = self.monkey_loc + self.monkey_img.get_height()/2

        # Move trees to the left, render and compute collision.
        self.next_tree -= self.horz_speed
        edge_hit = False
        tree_hit = False
        pass_tree = False
        for tree in self.trees:
            tree['x'] -= self.horz_speed

            if self.visual: 
                # Render tree.
                self.screen.blit(self.tree_img, (tree['x'], self.tree_offset))

                # Render gap in tree.
                self.screen.blit(self.background_img, (tree['x'], tree['y']),
                             (tree['x']-self.iter, tree['y'],
                              self.tree_img.get_width(), self.tree_gap))
                if self.iter < self.background_img.get_width() - self.screen_width:
                    self.screen.blit(self.background_img, (tree['x'], tree['y']),
                                 (tree['x']-(self.iter+self.background_img.get_width()), tree['y'],
                                  self.tree_img.get_width(), self.tree_gap))
                
            trunk_left  = tree['x'] + 215
            trunk_right = tree['x'] + 290
            trunk_top   = tree['y']
            trunk_bot   = tree['y'] + self.tree_gap

            # Compute collision.
            if (((trunk_left < (self.monkey_left+15)) and (trunk_right > (self.monkey_left+15))) or
                ((trunk_left < self.monkey_right) and (trunk_right > self.monkey_right))):
                #pg.draw.rect(self.screen, (255,0,0), (trunk_left, trunk_top, trunk_right-trunk_left, trunk_bot-trunk_top), 1)
                #pg.draw.rect(self.screen, (255,0,0), (self.monkey_left+15, monkey_top, self.monkey_img.get_width()-15, monkey_bot-monkey_top), 1)
                if (monkey_top < trunk_top) or (monkey_bot > trunk_bot):
                    tree_hit = True
            
            # Keep score.
            if not tree['s'] and (self.monkey_left+15) > trunk_right:
                tree['s'] = True
                self.score += 1
                pass_tree = True
                if self.sound:
                    self.blop_snd.play()

        # Monkey swings down on a vine.
        if self.vel < 0 and self.visual:
            pg.draw.line(self.screen, (92,64,51), (self.screen_width/2+20, self.monkey_loc-25), (self.hook,0), 4)

        # Render the monkey.
        if self.visual:
            self.screen.blit(self.monkey_img, (self.monkey_left, monkey_top))

        # Fail on hitting top or bottom.
        if monkey_bot > self.screen_height or monkey_top < 0:
            edge_hit = True

        # Render the score
        if self.visual:
            score_text = self.font.render("Score: %d" % (self.score), 1, (230, 40, 40))
            self.screen.blit(score_text, score_text.get_rect())

        if self.text is not None and self.visual:
            text = self.font.render(self.text, 1, (230, 40, 40))
            textpos = text.get_rect()
            self.screen.blit(text, (self.screen_width-textpos[2],0,textpos[2],textpos[3]))

        # Render the display.
        if self.visual: pg.display.update()

        # If failed, play sound and exit.  Also, assign rewards.
        if edge_hit:
            if self.sound:
                ch = self.screech_snd.play()
                while ch.get_busy():
                    pg.time.delay(500)
            if self.reward_fn is not None:
                self.reward_fn(self.edge_penalty)
            if self.action_fn is not None:
                self.action_fn(self.get_state())
            return False
        if tree_hit:
            if self.sound:
                ch = self.screech_snd.play()
                while ch.get_busy():
                    pg.time.delay(500)
            if self.reward_fn is not None:
                self.reward_fn(self.tree_penalty)
            if self.action_fn is not None:
                self.action_fn(self.get_state())
            return False

        if self.reward_fn is not None:
            if pass_tree:
                self.reward_fn(self.tree_reward)
            else:
                self.reward_fn(0.0)            
        
        # Wait just a bit.
        if self.visual: pg.time.delay(self.tick_length)

        # Move things.
        self.hook -= self.horz_speed
        self.iter -= self.horz_speed
        if self.iter < -self.background_img.get_width():
            self.iter += self.background_img.get_width()

        return True

# if called directly, creates a SwingyMonkey object with no learning
if __name__ == '__main__':
    
    # Create the game object.
    game = SwingyMonkey(visual=False)

    # Loop until you hit something.
    while game.game_loop():
        pass


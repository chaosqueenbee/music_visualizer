import librosa
import numpy as np
import pygame


def clamp(min_value, max_value, value):

    if value < min_value:
        return min_value

    if value > max_value:
        return max_value

    return value


class AudioBar:

    def __init__(self, x, y, freq, color, width=50, min_height=10, max_height=100, min_decibel=-80, max_decibel=0):

        self.x, self.y, self.freq = x, y, freq

        self.color = color

        self.width, self.min_height, self.max_height = width, min_height, max_height

        self.height = min_height

        self.min_decibel, self.max_decibel = min_decibel, max_decibel

        self.__decibel_height_ratio = (self.max_height - self.min_height)/(self.max_decibel - self.min_decibel)

    def update(self, dt, decibel):

        desired_height = decibel * self.__decibel_height_ratio + self.max_height

        speed = (desired_height - self.height)/0.1

        self.height += speed * dt

        self.height = clamp(self.min_height, self.max_height, self.height)

        self.color = (clamp(0, 255, self.height), 0, clamp(0, 255, self.height))

    def render(self, screen):

        pygame.draw.rect(screen, self.color, (self.x, self.y + self.max_height - self.height, self.width, self.height))


filename = "evolution.mp3"
n_fft = 2048 * 4

# Default sample rate is 22050, set optional sr to change that.
time_series, sample_rate = librosa.load(filename)

# Get amplitude values according to frequency and time indexes using short-time Fourier analysis
# See https://www.youtube.com/watch?v=T9x2rvdhaIE for context. (This is a matrix).
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=n_fft))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=n_fft)

# Get an array of time periodic
times = librosa.core.frames_to_time(np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=n_fft)

time_index_ratio = len(times)/times[len(times) - 1]

frequencies_index_ratio = len(frequencies)/frequencies[len(frequencies)-1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][int(target_time * time_index_ratio)]


pygame.init()

infoObject = pygame.display.Info()

screen_size = int(infoObject.current_w / 2)
screen = pygame.display.set_mode([screen_size, screen_size])
pygame.display.set_caption("Audio Visualizer")

bars = []

frequencies = np.arange(100, 5000, 100)
r = len(frequencies)
width_of_bar = screen_size/r
x = (screen_size - width_of_bar * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 300, c, (255, 0, 0), max_height=400, width=width_of_bar))
    x += width_of_bar

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(filename)
pygame.mixer.music.play(0)

# Run until the user asks to quit
running = True
while running:

    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    for b in bars:
        b.update(deltaTime, get_decibel(pygame.mixer.music.get_pos()/1000.0, b.freq))
        b.render(screen)

    # Flip the display
    pygame.display.flip()

pygame.quit()

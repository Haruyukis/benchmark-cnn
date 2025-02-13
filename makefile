# Définir le compilateur CUDA et le compilateur C
NVCC = nvcc
CC = gcc-10

# Drapeaux de compilation pour nvcc et gcc
CFLAGS = -lm -g
NVCCFLAGS = -ccbin $(CC) -g -Xcompiler -fPIC

# Dossier de sortie
BUILD_DIR = build

# Fichiers sources CUDA
CUDA_SRCS = src/main_fft.cu src/fft/fftShared.cu src/fft/ifftShared.cu \
            src/fft/fftSharedRow.cu src/shared/transpose.cu src/shared/hadamard.cu \
			src/shared/loadImageGPU.cu src/fft/convFFTShared.cu src/shared/storeImageGPU.cu

# Fichiers sources C
C_SRCS = src/shared/loadImage.c src/shared/storeImage.c

# Fichiers objets
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
C_OBJS = $(C_SRCS:.c=.o)

# Nom de l'exécutable
TARGET = $(BUILD_DIR)/mainFFT

# Règle par défaut (c'est-à-dire ce qui se passe quand on appelle `make`)
all: $(TARGET)

# Compilation des fichiers CUDA (.cu)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

# Compilation des fichiers C (.c)
%.o: %.c
	$(CC) -o $@ -c $<

# Lier les fichiers objets en un exécutable
$(TARGET): $(CUDA_OBJS) $(C_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CFLAGS)

# Nettoyage des fichiers objets et de l'exécutable
clean:
	rm -f $(CUDA_OBJS) $(C_OBJS) $(TARGET)

# Forcer la recompilation (utile pour forcer une recompilation complète)
.PHONY: all clean

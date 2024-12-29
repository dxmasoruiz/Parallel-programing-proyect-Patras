CC = gcc
CFLAGS = -Wall -O3 -fopenmp
TARGET = multistart_hooke_seq
OBJECTS = multistart_hooke_seq.o

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

multistart_hooke_seq.o: multistart_hooke_seq.c
	$(CC) $(CFLAGS) -c multistart_hooke_seq.c

clean:
	rm -f $(TARGET) $(OBJECTS)

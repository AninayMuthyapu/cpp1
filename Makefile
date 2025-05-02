# Compiler
CXX = g++
CXXFLAGS = -Wall -O2

# Find all .cpp files in the directory
SRCS = $(wildcard *.cpp)

# Convert .cpp to .o
OBJS = $(SRCS:.cpp=.o)

# Executable name = name of current directory
TARGET = $(notdir $(CURDIR))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJS)

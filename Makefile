CXX = g++
CXXFLAGS = -Wall -O2 -fopenmp

TARGET = cpp1.exe
SRCS = MatrixMulti1.cpp AnyOption/anyoption.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

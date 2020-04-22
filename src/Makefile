BILDDIR = ./build
OBJDIR = $(BUILDDIR)/obj
SRCDIR = ./

TARGET = $(BUILDDIR)/main

CXX = clang++ -std=c++17 
CXXFLAGS =  -O2 -Wall `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

SRCS := $(shell find $(SRCDIR) -name *.cpp)
OBJS := $(SRCS:%=$(OBJDIR)/%.o)
CPPFLAGS := $(INCLUDE) -MMD -MP
LDFLAGS += $(LIBDIRS) $(LIBS)


default:
	make all

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# c++ source
$(OBJDIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
	
.PHONY: all clean rebuild

clean:
	$(RM) -r $(BUILDDIR)

rebuild:
	make clean && make

-include $(DEPS)

MKDIR_P = mkdir -p

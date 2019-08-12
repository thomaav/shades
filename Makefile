CXX:=clang++

INC:=-Ivendor/glad/include/ -Ivendor/stb/include/

CFLAGS:=-Wall -Wextra -std=c++14 -g ${INC} -MD
LDLIBS:=-lfftw3 -lopenal -lalut -pthread -ldl -lglfw

OBJ_DIR:=objects
OBJS=$(patsubst %.cpp, $(OBJ_DIR)/%.o, $(wildcard src/*.cpp))
OBJS+=$(patsubst %.c, $(OBJ_DIR)/%.o, $(wildcard vendor/glad/src/*.c))

all: $(OBJS)
	${CXX} ${CFLAGS} $^ ${LDLIBS} -o shades

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	${CXX} ${CFLAGS} $< -c -o $@

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(@D)
	${CXX} ${CFLAGS} -x c++ $< -c -o $@

-include $(OBJS:.o=.d)

.PHONY: clean
clean:
	rm -rf objects
	rm shades

.PHONY: run
run: all
	@./shades


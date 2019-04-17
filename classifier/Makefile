CC = g++

Caffe_ROOT = /opt/caffe/ristretto

INCLUDE_DIR = $(Caffe_ROOT)/include -I/usr/local/cuda-9.0/include
LIBRARY_DIR = $(Caffe_ROOT)/build/lib -L/usr/local/cuda-9.0/lib64

#CFLAGS = -std=c++11 -DCPU_ONLY -DUSE_OPENCV
CFLAGS = -g -std=c++11 -DUSE_OPENCV
CFLAGS += -DUSE_LMDB -DUSE_LEVELDB

#LIBS = `pkg-config --libs opencv` -lboost_system -lglog -lcaffe
LIBS += -lstdc++ -lcblas -latlas -llmdb -lleveldb -lsnappy -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lm -lhdf5_hl -lhdf5
LIBS += -lboost_system -lglog -lcaffe

OPENCV = `pkg-config opencv --cflags --libs`
LIBS += $(OPENCV)

all: classifier

classifier: classifier.cpp
	$(CC) -o $@ $^ -I$(INCLUDE_DIR) $(CFLAGS) -L$(LIBRARY_DIR) $(LIBS)

clean:
	rm -f classifier


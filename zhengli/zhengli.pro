QT += core
QT -= gui

TARGET = zhengli
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    pix.cpp \
    kcf/fhog.cpp \
    kcf/kcftracker.cpp

INCLUDEPATH += E:\Opencv3\include
LIBS+= -L E:\Opencv3\lib\libopencv_*.a

HEADERS += \
    pix.h \
    kcf/ffttools.hpp \
    kcf/fhog.hpp \
    kcf/kcftracker.hpp \
    kcf/labdata.hpp \
    kcf/recttools.hpp \
    kcf/tracker.h

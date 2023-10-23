#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <ctime>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fstream>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

// Declaring global variables here or other useful stuff
pthread_mutex_t MutexFile;

// Defining needed Structures/Classes
struct Network
{
    int neurons;
    int layers;
    int neurons_per_layer;
};

/* This struct, SignalAndWait, provides a synchronization mechanism that can be used to signal and wait for
 a certain number of threads to complete their work. It contains a mutex lock and condition variable,
 as well as signal and limit counts. Threads can use the `signal()` function to signal that they have
 completed their work, and the `wait()` function can be used to wait for a specified number of signals
 to be received. This allows multiple threads to synchronize their work and ensure that they all complete
 before continuing to the next step in a program.*/
struct SignalAndWait
{

    pthread_mutex_t MutexWait;
    pthread_cond_t MutexSignal;
    int signalCount;
    int limitCount;

    // Constructor: initializes the mutex, condition variable, and signal and limit counts.
    SignalAndWait(int limit = 0)
    {
        pthread_mutex_init(&MutexWait, nullptr);
        pthread_cond_init(&MutexSignal, nullptr);
        signalCount = 0;
        limitCount = limit;
    }

    // Wait function: locks the mutex, waits for signals to reach the limit count, and unlocks the mutex.
    void wait()
    {
        pthread_mutex_lock(&MutexWait);
        while (signalCount != limitCount)
            pthread_cond_wait(&MutexSignal, &MutexWait);
        pthread_mutex_unlock(&MutexWait);
    }

    // Signal function: increments the signal count, signals the condition variable, and unlocks the mutex.
    void signal()
    {
        pthread_mutex_lock(&MutexWait);
        ++signalCount;
        pthread_cond_signal(&MutexSignal);
        pthread_mutex_unlock(&MutexWait);
    }

    // Destructor: destroys the mutex and condition variable.
    ~SignalAndWait()
    {
        pthread_mutex_destroy(&MutexWait);
        pthread_cond_destroy(&MutexSignal);
    }
};

// Hidden Layers have multiple attributes, so this struct rounds up all those attributes so that each hidden layer's
// attributes/variables can be accessed using 1 object only
struct HiddenLayer
{
    Network *neural = nullptr;
    SignalAndWait *synchronize;
    double **weight_matrix = nullptr;
    double *node = nullptr; // this variable takes input, basically input node from file
    int size = 0;

    void SetAll(Network *neural, SignalAndWait *synchronize, double **weight_matrix, double *node, int size)
    {
        this->neural = neural;
        this->synchronize = synchronize;
        this->weight_matrix = weight_matrix;
        this->node = node;
        this->size = size;
    }
};

// Following are the Function Protoypes
// receives the info and then performs on INPUT NODES
void ImplementInputFunctionality(Network *);
// function pointer accessed by a thread to process the hidden layer
void *HiddenLayerProcessing(void *);
// This is a function in which inner layers are being processed
void ImplementLayerFunctionality(double ***, Network *, int);
// This following function is the processing of input layer, meaning results are analysed here
void ImplementOutputFunctionality(double ***, Network *);

int main()
{
    pthread_mutex_init(&MutexFile, NULL);
    int n = 2;   // input node
    int l = 7;   // layers
    int npl = 8; // nodes pr layer
    Network net;
    net.neurons = n;
    net.layers = l;
    net.neurons_per_layer = npl;
    pid_t pid = fork();
    if (pid == 0) // child process basically does the processing for further layers
    {
        cout << flush;
        ImplementInputFunctionality(&net);
        exit(0);
    }
    else if (pid > 0)
    {
        // cout << getpid() << endl;
        int r;
        wait(&r);
        exit(0);
    }
}

// Following are the Function Definitions
void ImplementInputFunctionality(Network *a)
{
    // pipes for back propagation
    int GoBack1[2];
    pipe(GoBack1);
    int GoBack2[2];
    pipe(GoBack2);

    int NeuralNodes = a->neurons;
    pthread_t *tid = new pthread_t[NeuralNodes]; // implementing the functionality of each input neuron being used as a different thread
    SignalAndWait obj1(NeuralNodes);             // user defined lock mechanism to make sure all threads run
    double *input_node = new double[NeuralNodes]{0};
    double **weight = new double *[NeuralNodes];
    for (int i = 0; i < NeuralNodes; i++)
        *(weight + i) = new double[a->neurons_per_layer]{0};

    HiddenLayer hidden;
    hidden.neural = a;
    hidden.synchronize = &obj1;
    hidden.node = input_node;
    hidden.weight_matrix = weight;

    // using detached state so that i dont have to wait for the execution of each thread
    pthread_attr_t detachstate;
    pthread_attr_setdetachstate(&detachstate, PTHREAD_CREATE_DETACHED);
    for (int i = 0; i < NeuralNodes; i++)
        pthread_create(&(*(tid + i)), &detachstate, HiddenLayerProcessing, (void *)&hidden);

    // using my userdefined to function to wait for process to finish before ending the process
    obj1.wait();

    // now, moving on to creation of pipe for intraprocess communication
    double ***Propagate = new double **[a->neurons_per_layer];
    // The first dimension of the array corresponds to the number of neurons in the next layer,
    // the second dimension corresponds to the number of neurons in the input layer, and
    // the third dimension has a fixed size of two to represent the read and write file descriptors.
    for (int i = 0; i < a->neurons_per_layer; i++)
    {
        *(Propagate + i) = new double *[NeuralNodes];
        for (int j = 0; j < a->neurons_per_layer; j++)
        {
            *(*(Propagate + i) + j) = new double[2];
            double *localvar = *(*(Propagate + i) + j);
            pipe((int *)localvar);
        }
    }

    // moving onwards to the next layer
    pid_t pid = fork();
    if (pid == 0) // child process -> creates a layer
    {
        int fd = 0;
        ImplementLayerFunctionality(Propagate, a, fd);
        exit(0);
    }
    else
    {
        for (int i = 0; i < a->neurons_per_layer; i++)
        {
            int j = 0;
            while (j < NeuralNodes){ char buff[sizeof(double)]; *(double *)buff = hidden.weight_matrix[j][i];
                close(Propagate[i][j][0]); write(Propagate[i][j][1], buff, sizeof(buff));close(Propagate[i][j][1]);
            j++;}
        }

        // code for implementing back propagation
        double fx1 = 0; // temporary variable
        close(GoBack1[1]);
        char f1[sizeof(double)];
        read(GoBack1[0], f1, sizeof(double));
        fx1 = *(double *)f1;
        // cout << "fx(x1): " << fx1 << endl;

        double fx2 = 0; // temporary variable
        close(GoBack2[1]);
        char f2[sizeof(double)];
        read(GoBack2[0], f2, sizeof(double));
        fx2 = *(double *)f2;
        cout << "Fx(X2) = " << fx2 << endl;

        wait(NULL);
    }
}

void *HiddenLayerProcessing(void *args)
{
    HiddenLayer *hiddenlayer = (HiddenLayer *)args;
    // taking input from the configuration file
    pthread_mutex_lock(&MutexFile);
    ifstream file_object("file.txt");
    file_object >> hiddenlayer->node[hiddenlayer->size];

    // The reason is that the following loop is used is because each layer of the neural network has a different number of neurons, so the number of lines to be skipped
    //  in the input file is different for each layer. Therefore, the loop needs to read and discard the correct number of lines to ensure that the weights are read from the correct position in the file.
    string t;
    // now reading the weights
    for (int i = 0; i < hiddenlayer->size + 1; i++)
        getline(file_object, t);

    for (int i = 0; i < hiddenlayer->neural->neurons_per_layer; i++)
        file_object >> hiddenlayer->weight_matrix[hiddenlayer->size][i]; // hiddenlayer->size tells the layer number for each thread/layer

    // closing file and unlocking mutex
    file_object.close();
    pthread_mutex_unlock(&MutexFile);

    // performing a simple matrix multiplication
    for (int i = 0; i < hiddenlayer->neural->neurons_per_layer; i++)
        hiddenlayer->weight_matrix[hiddenlayer->size][i] *= hiddenlayer->node[hiddenlayer->size];

    hiddenlayer->synchronize->signal();
    hiddenlayer->size++; // move to the next layer
    pthread_exit(NULL);  // marking the end of the thread
}

void ImplementLayerFunctionality(double ***arr, Network *obj, int fd)
{
    // pipes for back propagation
    int GoBack1[2];
    pipe(GoBack1);
    int GoBack2[2];
    pipe(GoBack2);

    pthread_t *tid = new pthread_t[obj->neurons_per_layer]; // dynamically creating threads for each neuron in the hidden layer
    SignalAndWait lockmechanism(obj->neurons_per_layer);
    HiddenLayer *hidden = new HiddenLayer[obj->neurons_per_layer];
    int i = 0;
    while (i < obj->neurons_per_layer)
    {
        //sets all the values of HIDDEN LAYER STRUCTURE
        hidden[i].SetAll(obj, &lockmechanism, arr[i], new double[obj->neurons_per_layer], fd);
        pthread_create(&tid[i], NULL, HiddenLayerProcessing, (void *)&hidden[i]);
        i++;
    }

    // now, moving on to creation of pipe for intraprocess communication
    double ***Propagate = new double **[obj->neurons_per_layer];
    // The first dimension of the array corresponds to the number of neurons in the next layer,
    // the second dimension corresponds to the number of neurons in the input layer, and
    // the third dimension has a fixed size of two to represent the read and write file descriptors.
    for (int i = 0; i < obj->neurons_per_layer; i++)
    {
        *(Propagate + i) = new double *[obj->neurons_per_layer];
        for (int j = 0; j < obj->neurons_per_layer; j++)
        {
            *(*(Propagate + i) + j) = new double[2];
            double *localvar = *(*(Propagate + i) + j);
            pipe((int *)localvar);
        }
    }

    // moving onwards to the next layer
    pid_t pid = fork();
    if (pid == 0 && fd == obj->layers - 1) // if last layer has been processed, then OUTPUT F(X1) and F(X2)
    {
        ImplementOutputFunctionality(Propagate, obj);
        exit(0);
    }
    else if (pid == 0 && fd != obj->layers - 1) // child process -> creates a layer and base case for OUTPUT, need base case for recursion
    {
        // fd value is being updated in each recusive iteration
        ImplementLayerFunctionality(Propagate, obj, fd + 1);
        exit(0);
    }
    else // adding values in the next layer so that it can be processed. Happens OBJ->LAYER - 1 times
    {
        if (fd == obj->layers - 1)
        {
            int i = 0;
            while(i < obj->neurons_per_layer){
                // writes the outputs of the neurons in the last layer to the pipes
                char buff[sizeof(double)]; *(double *)buff = hidden[i].node[0];close(Propagate[i][0][0]); write(Propagate[i][0][1], buff, sizeof(buff));close(Propagate[i][0][1]); 
            i++;}
        }

        // BACK PROPAGATION
        close(GoBack1[0]);
        char f1[sizeof(double)];
        write(GoBack1[1], f1, sizeof(double));

        close(GoBack2[0]);
        char f2[sizeof(double)];
        write(GoBack2[1], f2, sizeof(double));

        for (int j = 0; j < obj->neurons_per_layer; j++)
        {
            int k = 0;
            while (k < obj->neurons_per_layer){
                // writes the outputs of the neurons in the last layer to the pipes
                char buff[sizeof(double)]; *(double *)buff = hidden[j].node[i]; close(Propagate[j][i][0]); write(Propagate[j][i][1], buff, sizeof(buff));close(Propagate[j][i][1]);
            k++;}
        }

        // BACK PROPAGATION
        close(GoBack1[0]); char f3[sizeof(double)]; write(GoBack1[1], f3, sizeof(double));

        close(GoBack2[0]); char f4[sizeof(double)]; write(GoBack2[1], f4, sizeof(double));

        wait(NULL);
    }
}

void ImplementOutputFunctionality(double ***arr, Network *obj)
{
    double x = 0;
    // readig values from the pipe, which is named as ARR in my case.
    // ARR is nothing but a pipe used for forward propagation.

    for (int i = 0; i < obj->neurons_per_layer; i++){close(arr[i][0][1]);char f1[sizeof(double)];read(arr[i][0][0], f1, sizeof(double));
        x += *(double *)f1;}

    double FX1 = 0;
    double FX2 = 0;

    FX1 = static_cast<double>(x * x + x + 1);
    FX2 = static_cast<double>((x * x - x) / 2);

    cout << "Fx(X1) = " << FX1 << endl;
    // cout << "Fx(X2) = " << FX2 << endl;
}

#include <iostream>
#include <cstdlib>

template<typename T>
class injector {
  public:
    injector() {
      srand(time(NULL));
      std::cout << "[i] initialized injector" << std::endl;
    };

    ~injector() {};

    void inject_random_error(T* input, size_t len) {
      int index = rand() % len;
      input[index] = 0;
    }

    void inject_random_error(T* input, size_t len, double probability) {
      double evt = (double)rand() / RAND_MAX;
      if(evt > probability)
      {
        inject_random_error(input, len);
      }
    }
};

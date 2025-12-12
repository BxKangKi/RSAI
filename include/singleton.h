#pragma once

template<typename T>
class singleton {
protected:
    singleton() = default;
public:
    static T& get() {
        static T instance;  // C++11+ thread-safe
        return instance;
    }
    // 복사/대입 방지 (singleton 클래스에 대해)
    singleton(const singleton&) = delete;
    singleton& operator=(const singleton&) = delete;
    singleton(singleton&&) = delete;
    singleton& operator=(singleton&&) = delete;
};

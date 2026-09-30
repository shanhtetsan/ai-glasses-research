#pragma once

#include <tiny_websockets/internals/ws_common.hpp>
#include <tiny_websockets/network/tcp_client.hpp>

namespace websockets { namespace network {
  template <class WifiClientImpl> 
  class GenericEspTcpClient : public TcpClient {
  public:
    GenericEspTcpClient(WifiClientImpl c) : client(c) {
      client.setNoDelay(true);
    }
    
    GenericEspTcpClient() {}

    bool connect(const WSString& host, const int port) {
      yield();
      auto didConnect = client.connect(host.c_str(), port);
      client.setNoDelay(true);
      return didConnect;
    }

    bool poll() {
      yield();
      return client.available();
    }

    bool available() override {
      return client.connected();
    }

    void send(const WSString& data) override {
      yield();
      writeOrClose(
        reinterpret_cast<const uint8_t*>(data.c_str()),
        static_cast<uint32_t>(data.size())
      );
      yield();
    }

    void send(const WSString&& data) override {
      yield();
      writeOrClose(
        reinterpret_cast<const uint8_t*>(data.c_str()),
        static_cast<uint32_t>(data.size())
      );
      yield();
    }

    void send(const uint8_t* data, const uint32_t len) override {
      yield();
      writeOrClose(data, len);
      yield();
    }
    
    WSString readLine() override {
      WSString line = "";

      int ch = -1;

      const uint64_t millisBeforeReadingHeaders = millis();
      while( ch != '\n' && available()) {
        if (millis() - millisBeforeReadingHeaders > _CONNECTION_TIMEOUT) return "";
        ch = client.read();
        if (ch < 0) continue;
        line += (char) ch;
      }

      return line;
    }

    uint32_t read(uint8_t* buffer, const uint32_t len) override {
      yield();
      return client.read(buffer, len);
    }

    void close() override {
      yield();
      client.stop();
    }

    virtual ~GenericEspTcpClient() {
      client.stop();
    }

  protected:
    void writeOrClose(const uint8_t* data, const uint32_t len) {
      const size_t written = client.write(data, len);
      if (written != len) {
        // A partial WebSocket frame cannot be resumed by the endpoint because
        // it submits each encoded frame with one transport write. Continuing
        // would make subsequent frames complete the wrong payload, so make the
        // failure observable and discard the TCP connection immediately.
        client.stop();
      }
    }

    WifiClientImpl client;

    int getSocket() const override {
      return -1;
    }    
  };
}} // websockets::network

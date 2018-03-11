#ifndef CBS_LOG_H
#define CBS_LOG_H

#include <string>
#include <sstream>


#define LOG_ERROR(x)							\
    {									\
	std::stringstream ss; 						\
	ss << "(" << __FILE__ << ", line " << __LINE__ << ") " << x;	\
	cbs::Log::error(ss.str());					\
    }

#define LOG_INFO(x)							\
    {									\
	std::stringstream ss; 						\
	ss << "(" << __FILE__ << ", line " << __LINE__ << ") " << x;	\
	cbs::Log::info(ss.str());					\
    }

#define LOG_WARN(x)							\
    {									\
	std::stringstream ss; 						\
	ss << "(" << __FILE__ << ", line " << __LINE__ << ") " << x;	\
	cbs::Log::warn(ss.str());					\
    }

#define LOG_ERROR_ONCE(x)							\
    {										\
	static bool not_done_yet = true;					\
	if (not_done_yet) {							\
	    std::stringstream ss;						\
	    ss << "(" << __FILE__ << ", line " << __LINE__ << ") " << x;	\
	    cbs::Log::error(ss.str());						\
	    not_done_yet = false;						\
	}									\
    }

#define LOG_INFO_ONCE(x)							\
    {										\
	static bool not_done_yet = true;					\
	if (not_done_yet) {							\
	    std::stringstream ss;						\
	    ss << "(" << __FILE__ << ", line " << __LINE__ << ") " << x;	\
	    cbs::Log::info(ss.str());						\
	    not_done_yet = false;						\
	}									\
    }

#define LOG_WARN_ONCE(x)							\
    {										\
	static bool not_done_yet = true;					\
	if (not_done_yet) {							\
	    std::stringstream ss;						\
	    ss << "(" << __FILE__ << ", line " << __LINE__ << ") " << x;	\
	    cbs::Log::warn(ss.str());						\
	    not_done_yet = false;						\
	}									\
    }

#define ASSERT(test)					        \
    {							        \
      if (!(test)) {					        \
        std::stringstream ss;				        \
	ss << "(" << __FILE__ << ", line" << __LINE__ << ") "   \
           << "assert \"" << #test << "\" failed";		\
        cbs::Log::error(ss.str());			        \
      }							        \
    }						        

#define TODO							\
    {							        \
        std::stringstream ss;				        \
	ss << "(" << __FILE__ << ", line" << __LINE__ << ") "   \
           << "function \"" << __FUNCTION__			\
           << "\" is not implemented yet";			\
	cbs::Log::error(ss.str());				\
    }						        



namespace cbs {
    class Log
    {
	enum Code {
	    FG_RED      = 31,
	    FG_GREEN    = 32,
	    FG_BLUE     = 34,
	    FG_YELLOW   = 33,
	    FG_DEFAULT  = 39,
	    BG_RED      = 41,
	    BG_GREEN    = 42,
	    BG_BLUE     = 44,
	    BG_DEFAULT  = 49
	};
    public:
	static bool use_exception;
	static bool use_color;

	static void error(const std::string& s);
	static void info(const std::string& s);
	static void warn(const std::string& s);
    };
    
}  // namespace cbs

#endif /* LOG_H */

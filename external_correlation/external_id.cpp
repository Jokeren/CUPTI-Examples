#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <atomic>

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
        __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestamp;

// Callback subscriber
static CUpti_SubscriberHandle cuptiSubscriber;

__thread int64_t localId;

  static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    default:
      break;
  }

  return "<unknown>";
}

  const char *
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "COMPILER";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "BUFFER_FLUSH";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "INSTRUMENTATION";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "RESOURCE";
    default:
      break;
  }

  return "<unknown>";
}

  const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
      return "PROCESS";
    case CUPTI_ACTIVITY_OBJECT_THREAD:
      return "THREAD";
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
      return "DEVICE";
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      return "CONTEXT";
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      return "STREAM";
    default:
      break;
  }

  return "<unknown>";
}

  uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
      return id->pt.processId;
    case CUPTI_ACTIVITY_OBJECT_THREAD:
      return id->pt.threadId;
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
      return id->dcs.deviceId;
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      return id->dcs.contextId;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      return id->dcs.streamId;
    default:
      break;
  }

  return 0xffffffff;
}

  static const char *
getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
      return "CUDA";
    case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
      return "CUDA_MPS";
    default:
      break;
  }

  return "<unknown>";
}

  static const char *
getStallReasonString(CUpti_ActivityPCSamplingStallReason reason)
{
  switch (reason) {
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID:
      return "Invalid";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE:
      return "Selected";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH:
      return "Instruction fetch";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY:
      return "Execution dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY:
      return "Memory dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE:
      return "Texture";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC:
      return "Sync";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY:
      return "Constant memory dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY:
      return "Pipe busy";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE:
      return "Memory throttle";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED:
      return "Not selected";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER:
      return "Other";
    default:
      break;
  }

  return "<unknown>";
}

  static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      {
        CUpti_ActivityExternalCorrelation *ec = (CUpti_ActivityExternalCorrelation *)record;
        printf("external id %d\n", ec->externalId);
        break;
      }
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
      {
        CUpti_ActivityPCSampling2 *psRecord = (CUpti_ActivityPCSampling2 *)record;

        printf("source %u, functionId %u, pc 0x%x, corr %u, samples %u, stallreason %s\n",
          psRecord->sourceLocatorId,
          psRecord->functionId,
          psRecord->pcOffset,
          psRecord->correlationId,
          psRecord->samples,
          getStallReasonString(psRecord->stallReason));
        break;
      }
    default:
      break;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  //free(buffer);
}


  void
finiTrace()
{
  CUPTI_CALL(cuptiUnsubscribe(cuptiSubscriber));
  CUPTI_CALL(cuptiEnableDomain(0, cuptiSubscriber, CUPTI_CB_DOMAIN_DRIVER_API));
  CUPTI_CALL(cuptiActivityFlushAll(0));
}


void cuptiSubscriberCallback(
  void *userdata,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId cb_id,
  const CUpti_CallbackData *cb_info
  )
{
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    if (cb_id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
      uint64_t id;
      if (cb_info->callbackSite == CUPTI_API_ENTER) {
        id = localId;
        printf("Push externalId %u\n", id);
        CUPTI_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, id));
      }
      if (cb_info->callbackSite == CUPTI_API_EXIT) {
        CUPTI_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, &id));
        printf("Pop externalId %u\n", id);
      }
    }
  }
}


  void
initTrace()
{
  // Subscribe callbacks
  CUPTI_CALL(cuptiSubscribe(&cuptiSubscriber,
      (CUpti_CallbackFunc) cuptiSubscriberCallback,
      (void *) NULL));
  CUPTI_CALL(cuptiEnableDomain(1, cuptiSubscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  size_t attrValue = 0, attrValueSize = sizeof(size_t);
  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  // Enable all other activity record kinds.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // Get and set activity attributes.
  // Attributes can be set by the CUPTI client to change behavior of the activity API.
  // Some attributes require to be set before any CUDA context is created to be effective,
  // e.g. to be applied to all device buffer allocations (see documentation).
  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));

  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

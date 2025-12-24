using System.Threading;

namespace  Unity.CognitionLearn
{
    internal static class MultiAgentGroupIdCounter
    {
        static int s_Counter;
        public static int GetGroupId()
        {
            return Interlocked.Increment(ref s_Counter);
        }
    }
}

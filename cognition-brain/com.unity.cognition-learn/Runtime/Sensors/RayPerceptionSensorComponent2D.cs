using UnityEngine;

namespace Unity.CognitionLearn.Sensors
{
    /// <summary>
    /// A component for 2D Ray Perception.
    /// </summary>
    [AddComponentMenu("Cognition Learn/Ray Perception Sensor 2D", (int)MenuGroup.Sensors)]
    public class RayPerceptionSensorComponent2D : RayPerceptionSensorComponentBase
    {
        /// <inheritdoc/>
        public override RayPerceptionCastType GetCastType()
        {
            return RayPerceptionCastType.Cast2D;
        }
    }
}

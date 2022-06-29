using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Carousel
{
    

public class PlayerControllerBase : MonoBehaviour
{
    public virtual bool IsDancing{  get { return false; }}
    public Transform root;
}

}